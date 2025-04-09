import sys
import glob
sys.path.append('/home/lihongxing/project/clu_mul')
import os
import datetime
os.environ["WANDB_MODE"] = "offline"
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils import checkpoint
from dateutil import tz
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin

from cgns.datasets.data_module import DataModule
from cgns.datasets.pretrain_dataset import (MultimodalPretrainingDataset,custom_collate_fn)
from cgns.datasets.transforms import DataTransforms
from cgns.models.backbones.encoder import ImageEncoder, BertEncoder

from cgns.models.mgca.utils.sentence_pool import SentenceAttentionPool
from cgns.models.backbones.image_decoder import ImageDecoder
from cgns.models.mgca.utils.local_attention import LocalCrossAttention
from cgns.models.mgca.utils.gather import SentenceGather
from cgns.models.mgca.utils.Kmeans_Cluster import kmeans_cluster

from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
# from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CGNS(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self, img_encoder='vit_base', stage1_epochs=10, stage2_epochs=10, stage1_warmup_epochs=1,
                 stage2_warmup_epochs=1, batch_size=4, gpus=None,
                 optim='adam', scheduler='linear_warmup_cosine_annealing', stage1_learning_rate=1e-5,
                 stage1_learning_rate_start=1e-7, stage1_learning_rate_end=0, stage1_weight_decay=1e-6,
                 stage2_learning_rate=1e-5, stage2_learning_rate_start=1e-7, stage2_learning_rate_end=0,
                 stage2_weight_decay=1e-6, temperature=0.1, emb_dim=768, k=128,
                 embed_dim=768, image_rec_drop_out_rate=0.5, gahter_pool='avg',
                 exclude_bn_bias=False, epsilon: float = 0.05, sinkhorn_iterations: int = 3,
                 proto_temperature: float = 0.2, freeze_prototypes_epochs: int = 1,
                 temp_decay='fixed', frozen_text_encoder=False, num_prototypes: int = 500, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        print(self.hparams)
        self.min_val_loss = float('inf')

        if img_encoder == "resnet_50":
            self.vision_width = 2048
        else:
            self.vision_width = 768
        self.text_width = 768 
        self.embed_dim = emb_dim

        # Define text global pooling over sentences
        self.global_text_attention = SentenceAttentionPool(16, self.embed_dim, pos_embed=False)

        # Define project
        self.local_vision_width = self.vision_width
        self.local_text_width = 768
        self.global_image_projection = nn.Linear(self.vision_width, self.embed_dim)
        self.local_image_projection = nn.Linear(self.local_vision_width, self.embed_dim)
        self.global_text_projection = nn.Linear(self.text_width, self.embed_dim)
        self.local_text_projection = nn.Linear(self.local_text_width, self.embed_dim)
        self.projector = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(self.embed_dim // 2,
                                                 self.embed_dim))
        self.image_decoder = ImageDecoder(embed_dim, encoder_name=img_encoder)
        # Define encoders
        self.image_encoder = ImageEncoder(
            model_name=img_encoder, pretrained=True)
        self.text_encoder = BertEncoder(pretrained=True)

        # Define local-interaction
        self.local_cross_attention = LocalCrossAttention(embed_dim)

        # Define temp for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        # Define hyper-params for optimization
        self.exclude_bn_bias = exclude_bn_bias  # False
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.stage1_warmup_epochs = stage1_warmup_epochs
        self.stage1_learning_rate = stage1_learning_rate
        self.stage1_learning_rate_start = stage1_learning_rate_start
        self.stage1_learning_rate_end = stage1_learning_rate_end
        self.stage1_weight_decay = stage1_weight_decay
        self.stage2_warmup_epochs = stage2_warmup_epochs
        self.stage2_learning_rate = stage2_learning_rate
        self.stage2_learning_rate_start = stage2_learning_rate_start
        self.stage2_learning_rate_end = stage2_learning_rate_end
        self.stage2_weight_decay = stage2_weight_decay
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.gpus = gpus
        self.epsilon = epsilon
        # self.lambd = lambd
        self.sinkhorn_iterations = sinkhorn_iterations
        self.proto_temperature = proto_temperature

        self.kc = kmeans_cluster(k)

        # Define NLP gather
        self.item_gather = SentenceGather(gahter_pool, embed_dim)

        # cache for loss
        self.last_local_batch_size = None
        self.global_alignment_labels = None

        # Define dataset
        self.train_dataset = MultimodalPretrainingDataset()
        # self.validation_dataset = validation_dataset
        # self.num_workers = num_workers
        self.train_iters_per_epoch = len(self.train_dataset) // (gpus * batch_size)

        # self.ckpt_path = ckpt_path
        self.best_path = self.hparams.best_path

        # freeze/finetuning params
        if frozen_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        self.inititalize_parameters()

        self.prototype_layer = nn.Linear(self.embed_dim, num_prototypes, bias=False)
        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn

        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(embed_dim // 2, embed_dim))

    def inititalize_parameters(self):
        # Initialize parameters
        nn.init.normal_(self.global_image_projection.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.global_text_projection.weight, std=self.text_width ** -0.5)
        nn.init.normal_(self.local_image_projection.weight, std=self.local_vision_width ** -0.5)
        nn.init.normal_(self.local_text_projection.weight, std=self.local_text_width ** -0.5)
        nn.init.normal_(self.projector[0].weight, std=self.embed_dim ** -0.5)

    def encode_image(self, image):
        # local_image_features, global_image_features, image_features_list = self.image_encoder(image)
        global_image_features,local_image_features = self.image_encoder(image)
        return self.local_image_projection(local_image_features), self.global_image_projection(global_image_features)

    def encode_text(self, text):
        x = self.text_encoder(text)
        # x = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        local_text_features = x['last_hidden_state']
        global_text_features = x['pooler_output']
        return self.local_text_projection(local_text_features), global_text_features

    def global_alignment_loss(self, image_embed, text_embed):
        # SimCLR style loss
        logit_scale = self.logit_scale.exp()
        local_batch_size = image_embed.size(0)
        if local_batch_size != self.last_local_batch_size:
            self.global_alignment_labels = local_batch_size * self.local_rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_embed_all = SyncFunction.apply(image_embed)
            text_embed_all = SyncFunction.apply(text_embed)
        else:
            image_embed_all = image_embed
            text_embed_all = text_embed

        # cosine similarity as logits (bs,768)
        logits_per_image = logit_scale * image_embed @ text_embed_all.contiguous().t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.contiguous().t()
        image_loss = F.cross_entropy(logits_per_image, self.global_alignment_labels)
        text_loss = F.cross_entropy(logits_per_text, self.global_alignment_labels)
        loss = (image_loss + text_loss) / 2
        global_alignment_loss = loss
        return global_alignment_loss, logits_per_image, logits_per_text

    def local_alignment_loss(self, local_image_embed_stacks, local_text_embed_stacks):
        total_image_loss = 0.
        text_to_local_image_embed_stacks = []
        image_to_local_text_embed_stacks = []
        # get each instance
        for idx in range(local_image_embed_stacks.size(0)):
            local_text_embed = local_text_embed_stacks[idx]
            local_image_embed = local_image_embed_stacks[idx]
            text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten = self.local_cross_attention(
                local_image_embed, local_text_embed)
            
            image_loss = self.simsiam_loss_func(local_image_embed, text_to_local_image_embed, self.predictor)
            total_image_loss += image_loss
            text_to_local_image_embed_stacks.append(text_to_local_image_embed.unsqueeze(0))
            image_to_local_text_embed_stacks.append(image_to_local_text_embed)
        # concatenate the text-to-image features to assist image reconstruction (under text condition)
        self.text_to_local_image_embed_stacks = torch.cat(text_to_local_image_embed_stacks, dim=0)  # (bs,49,768)
        # self.image_to_local_text_embed_stacks = torch.cat(image_to_local_text_embed_stacks, dim=0)
        local_image_loss = total_image_loss / local_image_embed_stacks.size(0)
        local_text_loss = self.text_local_loss_fn(local_text_embed_stacks, image_to_local_text_embed_stacks)
        # local_text_loss = (self.text_local_loss_fn(local_text_embed_stacks, image_to_local_text_embed_stacks, flag=1))
        return local_image_loss, local_text_loss

    def simsiam_loss_func(self, x, y, predictor, flag='image'):
        p_x = predictor(x)
        p_y = predictor(y)
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5

    def text_local_loss_fn(self, local_text_embed_stacks, image_to_local_text_embed_stacks, flag=1, kmeans_num=128,
                           temp=1.0,
                           kmean_cosine=0.4, bml_weight=1e-4):
        z1 = torch.cat(local_text_embed_stacks, dim=0)
        z2 = torch.cat(image_to_local_text_embed_stacks, dim=0)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        cos_sim = self.kc.sim(z1.unsqueeze(1), z2.unsqueeze(0))  

        fn_loss = None
        is_hard = False
        if kmeans_num > 0:
            normalized_cos = cos_sim * temp
            avg_cos = normalized_cos.mean().item()
            if not self.kc.initialized:
                if avg_cos <= kmean_cosine:
                    self.kc.optimized_centroid_init(z2)
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"kmeans start!!")
            elif self.kc.initialized:
                self.kc(z2, normalized_cos)
                is_hard = True
                z3, _, _ = self.kc.provide_hard_negative(z1)
                cos_sim_mask = self.kc.mask_false_negative(z1, normalized_cos)
                fn_loss = self.kc.false_negative_loss(z1, cos_sim_mask, normalized_cos, z3)
            self.kc.global_step += 1

            # Hard negative
            if is_hard == True:
                z1_z3_cos = self.kc.sim(z1.unsqueeze(1), z3.unsqueeze(0))  
                cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

                # Calculate loss with hard negatives
                # Note that weights are actually logits of weights
                z3_weight = self.kc.hard_negative_weight
                weights = torch.tensor(
                    [[1.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]).type_as(z1)

                cos_sim = cos_sim * weights

            labels = torch.arange(cos_sim.size(0)).type_as(z1).long()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cos_sim, labels)
            if fn_loss is not None:
                loss = loss + bml_weight * fn_loss
        return loss

    def recon_image(self, image_embed, image):
        # reshape the cross-modal features to the same shape as image
        # self.text_to_local_image_embed_stacks = self.text_to_local_image_embed_stacks.view(
        #     -1, self.text_to_local_image_embed_stacks.size(-1), *self.image_encoder.get_last_spatial_info())
        # # reshape the image features to the [B, C, H, W]
        # image_embed = image_embed.view(-1, image_embed.size(-1), *self.image_encoder.get_last_spatial_info())
        
        output = self.image_decoder(image_embed, self.text_to_local_image_embed_stacks, image)  # resnet:(bs,49,768) vit: ()
        rec_image_loss = output['loss']
        return rec_image_loss

    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:  # 遍历bs
            batch_stacks.append(
                self.global_text_attention(local_text_embed.unsqueeze(dim=0)))  # 取出每个句子计算全局文本注意力(sen_n,768)(1,768)
        return torch.cat(batch_stacks, dim=0)

    def sinkhorn(self, Q, nmb_iters):
        '''
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    #### Cross-model prototype alignment ####
    # def cpa(self, global_image_embed, global_text_embed, flag):
    #     with torch.no_grad():
    #         w = self.prototype_layer.weight.data.clone()
    #         w = F.normalize(w, dim=1, p=2)
    #         self.prototype_layer.weight.copy_(w)

    #     # Compute assign code of images and report
    #     img_proto_out = self.prototype_layer(global_image_embed)  # [bs,128]->[bs,500]图像和文本嵌入到原型空间
    #     report_proto_out = self.prototype_layer(global_text_embed)  # img_proto_out和report_proto_out分别代表图像和报告与原型的关联程度
    #     # ε的主要作用之一是控制除法和指数运算时的数值稳定性。较大的ε值可以减少因除法而导致的数值放大，从而避免exp函数中的输入值过大导致的溢出问题。但是，ε也不能太大，否则会过分抑制原型分配的差异性，影响模型学习能力。
    #     # TODO: define this to hparams
    #     with torch.no_grad():  # 使用Sinkhorn算法来获得每种原型与图像或报告样本之间的“分配码”
    #         img_code = torch.exp(
    #             img_proto_out / self.epsilon).t().contiguous()  # (bs,500)->[500,bs] 表示每种原型商品与批次中每个顾客需求之间的关联度量
    #         img_code = self.get_assignments(
    #             img_code,
    #             self.sinkhorn_iterations)  # 用sinkhorn算法得到关系度量，可以看作是从样本到原型的一种“运输计划”，表明如何将顾客需求“分配”给不同的原型商品
    #         report_code = torch.exp(
    #             report_proto_out / self.epsilon).t().contiguous()
    #         report_code = self.get_assignments(
    #             report_code, self.sinkhorn_iterations)  # bz, 500

    #     # 对分配码应用softmax函数，得到概率分布，就是软聚类分配码q
    #     img_proto_prob = F.softmax(
    #         img_proto_out / self.proto_temperature, dim=1)
    #     report_proto_prob = F.softmax(
    #         report_proto_out / self.proto_temperature, dim=1)

    #     # 计算图像到报告的对比损失，用负对数似然函数
    #     loss_i2t_proto = - \
    #         torch.mean(
    #             torch.sum(img_code * torch.log(report_proto_prob),
    #                       dim=1))  # 衡量了图像的分配码与报告的原型概率分布之间的匹配程度，表示图像和报告在原型空间中的对应更加一致，即它们的信息内容更加相关。
    #     loss_t2i_proto = - \
    #         torch.mean(torch.sum(report_code *
    #                              torch.log(img_proto_prob), dim=1))

    #     loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.  # L_CPA
    #     cap_loss = loss_proto
    #     return cap_loss

    def stage1_step(self, batch, split="train"):
        image = batch['image']  # (bs,3,224,224)
        text = batch['text']  # input_id:(bs,256)
        '''
        =================================================================
        Encode image and text and get the local and global representation
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # local_image_embed, global_image_embed = checkpoint(self.encode_image, image)
        local_image_embed = F.normalize(local_image_embed, dim=-1)
        global_image_embed = F.normalize(global_image_embed, dim=-1)
        # image_atts = torch.ones(local_image_embed.size()[:-1], dtype=torch.long).to(image.device)

        # Embed text
        local_text_embed, _ = self.encode_text(text)
        # local_text_embed, _ = checkpoint(self.encode_text, text)
        # local_text_embed, _ = checkpoint(self.encode_text, text, preserve_rng_state=True, use_reentrant=False)
        local_text_embed = F.normalize(local_text_embed, dim=-1)
        # gather local text embedding on sentence level
        local_text_embed_stacks,_ = self.item_gather(local_text_embed, batch)
        # get global text embedding  
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)
        global_text_embed = F.normalize(global_text_embed, dim=-1)

        '''
        =================================================================
        Calculate the alignment loss
        =================================================================
        '''

        # local_image_loss, local_text_loss = self.local_alignment_loss(local_image_embed, padding_local_text_embed)
        local_image_loss, local_text_loss = self.local_alignment_loss(local_image_embed, local_text_embed_stacks)
        # global contrastive loss
        global_alignment_loss, logits_per_image, logits_per_text = self.global_alignment_loss(global_image_embed,
                                                                                              global_text_embed)
        bz = global_image_embed.size(0)
        labels = torch.arange(bz).type_as(global_text_embed).long()
        
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            logits_per_image, labels, top_k=(1, 5))  
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            logits_per_text, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        # '''
        # =================================================================
        # Calculate cap loss
        # =================================================================
        # '''
        # cpa_loss = self.cpa(global_image_embed, global_text_embed, flag='stage1')

        '''
        =================================================================
        Log the loss
        =================================================================
        '''
        if split == 'train':
            loss_dict = {
                "stage1_local_image_loss": local_image_loss,
                "stage1_local_text_loss": local_text_loss,
                "stage1_global_alignment_loss": global_alignment_loss,
                # "stage1_cpa_loss": cpa_loss,
                "stage1_acc1": acc1,
                "stage1_acc5": acc5
                # "stage1_mlm_loss": loss_mlm
            }
            loss = local_image_loss + local_text_loss + global_alignment_loss + cpa_loss
            loss_dict["stage1_loss"] = loss
            return local_image_embed, loss_dict
        else:
            loss_dict = {
                "val_stage1_local_image_loss": local_image_loss,
                "val_stage1_local_text_loss": local_text_loss,
                "val_stage1_global_alignment_loss": global_alignment_loss,
                # "val_stage1_cpa_loss": cpa_loss,
                "val_stage1_acc1": acc1,
                "val_stage1_acc5": acc5
                # "val_stage1_mlm_loss": loss_mlm
            }
            loss = local_image_loss + local_text_loss + global_alignment_loss
            loss_dict["val_stage1_loss"] = loss
            return local_image_embed, loss_dict

    def stage2_step(self, batch, split="train"):
        '''
        Stage 3 employs the alignment with SPB and CCR
        '''
        image = batch['image']
        text = batch['text']
        '''
        =================================================================
        Encode image and text and get the local and global representation
        In stage 3, we need SPB to get the sentence prototype
        ================================================================='token_type_ids' = {Tensor: (10, 256)} tensor([[0, 0, 0,  ..., 0, 0, 0],\n        [0, 0, 0,  ..., 0, 0, 0],\n        [0, 0, 0,  ..., 0, 0, 0],\n        ...,\n        [0, 0, 0,  ..., 0, 0, 0],\n        [0, 0, 0,  ..., 0, 0, 0],\n        [0, 0, 0,  ..., 0, 0, 0]], device='cuda:0')
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)
        # local_image_embed, global_image_embed = checkpoint(self.encode_image, image)
        local_image_embed = F.normalize(local_image_embed, dim=-1)
        global_image_embed = F.normalize(global_image_embed, dim=-1)
        # image_atts = torch.ones(local_image_embed.size()[:-1], dtype=torch.long).to(image.device)

        # Embed text
        local_text_embed, _ = self.encode_text(text)
        # local_text_embed, _ = checkpoint(self.encode_text, text, preserve_rng_state=True, use_reentrant=False)
        local_text_embed = F.normalize(local_text_embed, dim=-1)
        # gather local text embedding on sentence level
        local_text_embed_stacks,_ = self.item_gather(local_text_embed, batch)
        # get global text embedding
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)
        global_text_embed = F.normalize(global_text_embed, dim=-1)

        '''
        =================================================================
        Calculate the alignment loss (LAM)

        =================================================================
        '''
        # global contrastive loss
        global_alignment_loss, logits_per_image, logits_per_text = self.global_alignment_loss(global_image_embed,
                                                                                              global_text_embed)
        bz = global_image_embed.size(0)
        labels = torch.arange(bz).type_as(global_text_embed).long()
        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            logits_per_image, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            logits_per_text, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        # local alignment loss: local_image_loss,local_text_loss
        local_image_loss, local_text_loss = self.local_alignment_loss(local_image_embed,
                                                                      local_text_embed_stacks)

        # '''
        # =================================================================
        # Calculate cap loss
        # =================================================================
        # '''
        # cpa_loss = self.cpa(global_image_embed, global_text_embed, flag='stage2')
        '''
        =================================================================
        Conditional Cross-modality reconstruction loss (CCR)
        =================================================================
        '''
        # Reconstruct image
        rec_image_loss = self.recon_image(local_image_embed, image)  # resnet:(bs,49,768),vit:(bs,196,768)
        '''
        =================================================================
        Log the loss
        =================================================================
        '''
        # 灵活地管理和调整不同组件的损失贡献，特别是通过调整乘数（如全局对齐损失乘以10）来平衡不同损失项对最终优化目标的影响
        if split == 'train':
            loss_dict = {
                "stage2_local_image_loss": local_image_loss,  # -0.0079
                "stage2_local_text_loss": local_text_loss,  # 2.7833
                "stage2_global_alignment_loss": global_alignment_loss,  # 2.8035
                # "stage2_cpa_loss": cpa_loss,  # 6.4099
                "stage2_rec_image_loss": rec_image_loss,  # 0.5812
                "stage2_acc1": acc1,
                "stage2_acc5": acc5
                # "stage2_mlm_loss": loss_mlm
            }
            loss = (local_image_loss + local_text_loss + global_alignment_loss + rec_image_loss)
            loss_dict["stage2_loss"] = loss

            return local_image_embed, loss_dict
        else:
            loss_dict = {
                "val_stage2_local_image_loss": local_image_loss,  # -0.0079
                "val_stage2_local_text_loss": local_text_loss,  # 2.7833
                "val_stage2_global_alignment_loss": global_alignment_loss,  # 2.8035
                # "val_stage2_cpa_loss": cpa_loss,  # 6.4099
                "val_stage2_rec_image_loss": rec_image_loss,  # 0.5812
                "val_stage2_acc1": acc1,
                "val_stage2_acc5": acc5
                # "val_mlm_loss": loss_mlm
            }
            loss = (local_image_loss + local_text_loss + global_alignment_loss + rec_image_loss)
            loss_dict["val_stage2_loss"] = loss
            return local_image_embed, loss_dict

    def forward(self, batch):
        image_features = self.image_encoder(batch)
        text_features = self.text_encoder(batch)
        return image_features, text_features

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            # Stage 1 starts
            optimizers, lr_schedulers, _ = self.call_optimization(max_epochs=self.stage1_epochs,
                                                                  warmup_epochs=self.stage1_warmup_epochs,
                                                                  weight_decay=self.stage1_weight_decay,
                                                                  learning_rate=self.stage1_learning_rate,
                                                                  learning_rate_start=self.stage1_learning_rate_start,
                                                                  learning_rate_end=self.stage1_learning_rate_end)
            self.trainer.lr_schedulers = lr_schedulers
            self.trainer.optimizers = optimizers
        elif self.current_epoch == self.stage1_epochs:
            # Load the best model from stage 1
            checkpoint_pattern = os.path.join(self.best_path, "epoch=*-step=*.ckpt")
            checkpoint_files = glob.glob(checkpoint_pattern)
            if checkpoint_files:
                # Sort by epoch and step to get the best model
                checkpoint_files.sort(
                    key=lambda x: (int(x.split('=')[1].split('-')[0]), int(x.split('=')[2].split('.')[0])))
                checkpoint_path = checkpoint_files[-1]  # Get the last (best) checkpoint
                if os.path.exists(checkpoint_path):
                    self.load_from_checkpoint(checkpoint_path)
                    print("finished load stage1_best {}".format(checkpoint_path))
            else:
                print("unfinished load stage1 best model.")

            optimizers, lr_schedulers, _ = self.call_optimization(max_epochs=self.stage2_epochs,
                                                                  warmup_epochs=self.stage2_warmup_epochs,
                                                                  weight_decay=self.stage2_weight_decay,
                                                                  learning_rate=self.stage2_learning_rate,
                                                                  learning_rate_start=self.stage2_learning_rate_start,
                                                                  learning_rate_end=self.stage2_learning_rate_end)
            self.trainer.lr_schedulers = lr_schedulers
            self.trainer.optimizers = optimizers

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.stage1_epochs - 1:
            if self.global_rank == 0:
                self.trainer.save_checkpoint(f"{self.best_path}/stage1_end.ckpt")

        if self.current_epoch > self.stage1_epochs - 1:
            # self.min_val_loss = float('inf')
            current_val_loss = self.trainer.callback_metrics['val_loss'].item()
            if current_val_loss < self.min_val_loss:
                self.min_val_loss = current_val_loss
                print(f"new_val_loss:{self.min_val_loss}")
                if self.global_rank == 0:
                    self.trainer.save_checkpoint(f"{self.best_path}/best_model_stage2.ckpt")

    def call_optimization(self, max_epochs=None, warmup_epochs=None, learning_rate=None, learning_rate_start=None,
                          learning_rate_end=None, weight_decay=None, slow_text_encoder=False):
        optim_conf = self.configure_optimizers(max_epochs=max_epochs, warmup_epochs=warmup_epochs,
                                               slow_text_encoder=slow_text_encoder, learning_rate=learning_rate,
                                               learning_rate_start=learning_rate_start,
                                               learning_rate_end=learning_rate_end, weight_decay=weight_decay)
        optimizers, lr_schedulers, optimizer_frequencies, monitor = self.trainer._configure_optimizers(optim_conf)
        lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
        return optimizers, lr_schedulers, optimizer_frequencies

    def training_step(self, batch,
                      batch_idx):
        if self.current_epoch < self.stage1_epochs:
            image_feaures, loss_dict = self.stage1_step(batch)
            loss = loss_dict['stage1_loss']
        elif self.current_epoch >= self.stage1_epochs:
            image_feaures, loss_dict = self.stage2_step(batch)
            loss = loss_dict['stage2_loss']
        self.log_dict(loss_dict, on_step=True, on_epoch=True, sync_dist=True,
                      prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.stage1_epochs:
            image_feaures, loss_dict = self.stage1_step(batch, split="val")
            loss = val_loss = loss_dict['val_stage1_loss']
        elif self.current_epoch >= self.stage1_epochs:
            image_feaures, loss_dict = self.stage2_step(batch, split="val")
            loss = val_loss = loss_dict['val_stage2_loss']
        loss_dict['val_loss'] = val_loss
        self.log_dict(loss_dict, on_step=True, on_epoch=True, sync_dist=True,
                      prog_bar=True) 

        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def exclude_from_text_encoder(self, named_params, weight_decay):
        # exclude discriminator param
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif 'text_encoder' in name:
                excluded_params.append(param)
            else:
                params.append(param)
        return params, excluded_params  

    def configure_optimizers(self, learning_rate=1e-5, learning_rate_start=1e-7, learning_rate_end=0, max_epochs=40,
                             warmup_epochs=1, slow_text_encoder=False, weight_decay=1e-6):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=weight_decay)
        else:
            params = self.parameters()
        if slow_text_encoder:  
            other_params, text_params = self.exclude_from_text_encoder(self.named_parameters(),
                                                                       weight_decay=weight_decay)
            params = [{"params": text_params}, {"params": other_params}]
        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif self.optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        if slow_text_encoder:
            optimizer.param_groups[0]['lr'] = learning_rate / 10
            optimizer.param_groups[1]['lr'] = learning_rate
        warmup_steps = self.train_iters_per_epoch * warmup_epochs
        total_steps = self.train_iters_per_epoch * max_epochs
        if self.scheduler == 'cosine_warmup_linear_annealing':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'linear_warmup_cosine_annealing':
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=warmup_steps,
                    max_epochs=total_steps,
                    warmup_start_lr=learning_rate_start, eta_min=learning_rate_end),
                "interval": "step",
                "frequency": 1,
            }
        elif self.scheduler == 'cosine_decay':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps),
                "interval": "step",
                "frequency": 1,
            }
        return [optimizer], [scheduler]

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="resnet_50")
        parser.add_argument("--freeze_bert", default=False, action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=768, help="768")
        parser.add_argument("--num_workers", type=int, default=20)

        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                            help='weight on off-diagonal terms')
        parser.add_argument("--stage1_epochs", type=int, default=10)
        parser.add_argument("--stage2_epochs", type=int, default=20)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--data_pct", type=float, default=1.)
        parser.add_argument("--k", type=int, default=128)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()  
    parser = Trainer.add_argparse_args(parser)  
    parser = CENSJL.add_model_specific_args(parser)
    args = parser.parse_args()
    args.deterministic = True
    seed_everything(args.seed)
    args.max_epochs = args.stage1_epochs + args.stage2_epochs
    args.precision = 16
    # debug
    # args.limit_train_batches = 0.1
    # args.limit_val_batches = 0.1
    args.num_sanity_val_steps = 0

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    args.best_path = ckpt_dir

    # Add load from checkpoint
    model = CENSJL(**args.__dict__)  
    should_resume = False
    if should_resume:
        args.resume_from_checkpoint = "/home/lihongxing/project/MGCA/data/ckpts/MGCA/2024_08_03_14_18_36/last.ckpt"
        model.kc.initialized = True

    datamodule = DataModule(MultimodalPretrainingDataset, custom_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    callbacks = [
        LearningRateMonitor(logging_interval="step"), 
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1)
        # EarlyStopping(monitor="val_loss", min_delta=0.00001,
        #               patience=10, verbose=True, mode="min")
    ]

    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    # os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="CENS", save_dir=logger_dir, name=extension)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,  
        logger=wandb_logger) 

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
