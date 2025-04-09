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
from dateutil import tz  # 提供一些处理时区的功能
from pytorch_lightning import LightningModule, Trainer, seed_everything  # 轻量级的PyTorch库
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint, Callback)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin

from mgca.datasets.data_module import DataModule  # DataModule_pretrain
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,custom_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import ImageEncoder, BertEncoder  # BertEncoder替换了
# from mgca.models.mgca.utils.xbert import BertConfig, BertForMaskedLM
# from mgca.models.mgca.utils.tokenization_bert import BertTokenizer

from mgca.models.mgca.utils.sentence_pool import SentenceAttentionPool
from mgca.models.backbones.image_decoder import ImageDecoder
from mgca.models.mgca.utils.local_attention import LocalCrossAttention
from mgca.models.mgca.utils.gather import SentenceGather
from mgca.models.mgca.utils.Kmeans_Cluster import kmeans_cluster

from pl_bolts.models.self_supervised.simclr.simclr_module import SyncFunction
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
# from torch.utils.checkpoint import checkpoint

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# 控制是否启用cuDNN的自动调优功能，默认值是Faulse,意味着PyTorch会根据网络的输入选择一个固定的最佳算法；
# 为True那么PyTorch会在网络的每个卷积层中，动态地寻找当前输入下最快的卷积算法，这样可以提高网络的运行效率，尤其是当网络的输入形状不固定时。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CENSJL(LightningModule):
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

        self.save_hyperparameters()  # 在 LightningModule 中定义模型时，调用这个方法可以自动将类初始化时传入的参数（通过 __init__ 方法）保存起来。这些超参数会被保存在 hparams 属性中，并且可以在训练、测试、验证等阶段访问，同时也方便日志记录、模型恢复和实验复现。
        print(self.hparams)
        self.min_val_loss = float('inf')
        # mlm_config = {"hidden_size": 768, "vocab_size": 28996, "hidden_act": 'gelu', "layer_norm_eps": 1e-12,
        #               'is_cross_attention': False, "encoder_width": 768}
        # bert_config = BertConfig.from_dict(mlm_config)

        # Get embedding space from vision/language model
        if img_encoder == "resnet_50":
            self.vision_width = 2048  # 图片输出维度
        else:
            self.vision_width = 768
        self.text_width = 768  # 文本特征的维度
        self.embed_dim = emb_dim  # 图片和文本特征统一的维度大小

        # Define text global pooling over sentences
        self.global_text_attention = SentenceAttentionPool(16, self.embed_dim, pos_embed=False)  # 一个文本16个句子

        # Define project
        self.local_vision_width = self.vision_width  # 局部图片特征维度
        self.local_text_width = 768  # 局部文本特征维度
        self.global_image_projection = nn.Linear(self.vision_width, self.embed_dim)  # 2048/512->768
        self.local_image_projection = nn.Linear(self.local_vision_width, self.embed_dim)  # ->768
        self.global_text_projection = nn.Linear(self.text_width, self.embed_dim)  # 768->768
        self.local_text_projection = nn.Linear(self.local_text_width, self.embed_dim)  # 768->768
        self.projector = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(self.embed_dim // 2,
                                                 self.embed_dim))  # output layer # used for simsiam loss
        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(768, affine=False)
        # self.ln = nn.LayerNorm(768)

        # # Define decoders
        # self.num_queries = num_queries  # 每个文本规定16个句子
        # self.image_decoder = ImageDecoder(embed_dim * 2, encoder_name=img_encoder,
        #                                   image_dropout=image_rec_drop_out_rate)
        self.image_decoder = ImageDecoder(embed_dim, encoder_name=img_encoder)
        # Define encoders
        self.image_encoder = ImageEncoder(
            model_name=img_encoder, pretrained=True)  # hparams是一个字典，包含一些超参数的值
        self.text_encoder = BertEncoder(pretrained=True)

        # path = "/home/lihongxing/project/PRIOR-main/codes/prior/encoders/language/Bio_ClinicalBERT"
        # self.text_encoder = BertForMaskedLM.from_pretrained(path, config=bert_config)  # BertEncoder, BertForMaskedLM
        # self.mlm_probability = 0.15
        # self.tokenizer = BertTokenizer.from_pretrained(path)

        # Define local-interaction
        self.local_cross_attention = LocalCrossAttention(embed_dim)

        # Define temp for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        # self.local_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / local_temperature))

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

        self.kc = kmeans_cluster(k)  # 实例化聚类

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

        # self.prototype_layer = nn.Linear(self.embed_dim, num_prototypes, bias=False)
        # if self._use_ddp_or_dpp2(self.trainer):
        #     self.get_assignments = self.distributed_sinkhorn
        # else:
        #     self.get_assignments = self.sinkhorn
            #  原型向量的维度,num_prototypes 这个变量的值，它表示了有多少个不同的类别。
            # 分配函数的作用是计算输入的特征向量和输出的原型向量之间的相似度（similarity），并根据相似度的大小，把特征向量分配到最接近的原型向量所代表的类别。

        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(embed_dim // 2, embed_dim))  # output layer # used for simsiam loss

    def inititalize_parameters(self):  # 以正态分布的方式对以下几个层的权重进行初始化
        # Initialize parameters
        nn.init.normal_(self.global_image_projection.weight, std=self.vision_width ** -0.5)
        nn.init.normal_(self.global_text_projection.weight, std=self.text_width ** -0.5)
        nn.init.normal_(self.local_image_projection.weight, std=self.local_vision_width ** -0.5)
        nn.init.normal_(self.local_text_projection.weight, std=self.local_text_width ** -0.5)
        nn.init.normal_(self.projector[0].weight, std=self.embed_dim ** -0.5)

    def encode_image(self, image):
        # local_image_features, global_image_features, image_features_list = self.image_encoder(image)
        global_image_features,local_image_features = self.image_encoder(image)  # (bs,2048/768) (bs,49,2048/768)
        return self.local_image_projection(local_image_features), self.global_image_projection(global_image_features)
        # (64,49,2048)->(64,49,768)  (64,2048)->(64,768)

    def encode_text(self, text):
        x = self.text_encoder(text)
        # x = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        local_text_features = x['last_hidden_state']  # (bs,256,768) 256：文本中的单词最多256
        global_text_features = x['pooler_output']  # (bs,768)Although we get the global features, we do not use it
        return self.local_text_projection(local_text_features), global_text_features

    def global_alignment_loss(self, image_embed, text_embed):
        # SimCLR style loss
        logit_scale = self.logit_scale.exp()
        local_batch_size = image_embed.size(0)
        if local_batch_size != self.last_local_batch_size:  # 根据当前批次大小更新全局对齐标签,确保每个样本有唯一的全局索引
            self.global_alignment_labels = local_batch_size * self.local_rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size  # 记录上一批次的大小，以便下次迭代时判断是否需要重新生成全局对齐标签

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)  # L2归一化，使得它们具有单位长度，有利于计算余弦相似度
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs  记得调试一下分布式
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            image_embed_all = SyncFunction.apply(image_embed)  # 同步在各个GPU上分别计算得到的特征向量,使得所有进程都能获得全局一致的、合并后的特征集合
            text_embed_all = SyncFunction.apply(text_embed)
        else:
            image_embed_all = image_embed
            text_embed_all = text_embed

        # cosine similarity as logits (bs,768)
        logits_per_image = logit_scale * image_embed @ text_embed_all.contiguous().t()  # 在一个bs内计算每幅图像与所有文本特征的余弦相似度得分
        logits_per_text = logit_scale * text_embed @ image_embed_all.contiguous().t()  # 每段文本与所有图像特征的余弦相似度得分
        image_loss = F.cross_entropy(logits_per_image, self.global_alignment_labels)
        text_loss = F.cross_entropy(logits_per_text, self.global_alignment_labels)
        loss = (image_loss + text_loss) / 2
        global_alignment_loss = loss  # 调整3不行
        return global_alignment_loss, logits_per_image, logits_per_text

    def local_alignment_loss(self, local_image_embed_stacks, local_text_embed_stacks):
        total_image_loss = 0.
        text_to_local_image_embed_stacks = []
        image_to_local_text_embed_stacks = []
        # TODO: maybe we can optimize this step ?
        # get each instance
        for idx in range(local_image_embed_stacks.size(0)):
            local_text_embed = local_text_embed_stacks[idx]  # (num_sen,768)
            local_image_embed = local_image_embed_stacks[idx]  # (49,768)
            text_to_local_image_embed, text_to_local_image_atten, image_to_local_text_embed, image_to_local_text_atten = self.local_cross_attention(
                local_image_embed, local_text_embed)
            # for local text-to-image alignment, we employ the simsiam loss without negative sample
            # image_loss = self.BarlowTwins_loss_fun(local_image_embed, text_to_local_image_embed)
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

    def simsiam_loss_func(self, x, y, predictor, flag='image'):  # 理论取值范围是(-1,0),越接近-1越好
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

        # if flag == 1:
        #     kc = self.kc1
        # elif flag == 2:
        #     kc = self.kc2
        # else:
        #     raise ValueError("Invalid flag value. Expected 1 or 2.")

        cos_sim = self.kc.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # z1中的每一个特征向量与z2中的每一个特征向量之间的余弦相似度(160,160)
        # cos_sim = torch.bmm(z1, z2.permute(0,2,1))

        fn_loss = None
        is_hard = False
        if kmeans_num > 0:
            normalized_cos = cos_sim * temp
            avg_cos = normalized_cos.mean().item()
            if not self.kc.initialized:
                if avg_cos <= kmean_cosine:  # 是否是开始K-means聚类的好时机。如果数据点间的平均相似度较低，表明数据较为分散，适合进行聚类划分
                    self.kc.optimized_centroid_init(z2)  # 从z2中选择最不相关的作为初始化质心
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"kmeans start!!")
            elif self.kc.initialized:
                # if cls.cluster.global_step % 100 == 0:
                #     if not dist.is_initialized() or dist.get_rank() == 0:
                #         print(cls.cluster.centroid.data[0][:4].tolist())
                self.kc(z2, normalized_cos)  # 更新了聚类中心
                is_hard = True  # to be fix
                z3, _, _ = self.kc.provide_hard_negative(z1)  # 得到硬负样本
                cos_sim_mask = self.kc.mask_false_negative(z1, normalized_cos)
                fn_loss = self.kc.false_negative_loss(z1, cos_sim_mask, normalized_cos, z3)
            self.kc.global_step += 1

            # Hard negative
            if is_hard == True:
                z1_z3_cos = self.kc.sim(z1.unsqueeze(1), z3.unsqueeze(0))  # 引入额外的负样本，特别是硬负样本，以增强模型的学习。
                cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)  # to be fix 拼接相似度矩阵(256,512)，添加负样本的数量

                # Calculate loss with hard negatives
                # Note that weights are actually logits of weights
                z3_weight = self.kc.hard_negative_weight  # 为硬负样本分配一个权重,通常这个权重会比普通负样本大
                weights = torch.tensor(
                    [[1.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (
                            z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]).type_as(z1)

                cos_sim = cos_sim * weights

            labels = torch.arange(cos_sim.size(0)).type_as(z1).long()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cos_sim, labels)  # 普通的对比损失
            if fn_loss is not None:
                loss = loss + bml_weight * fn_loss
            # print(loss)
        return loss

    # def off_diagonal(self, x):
    #     # return a flattened view of the off-diagonal elements of a square matrix
    #     n, m = x.shape
    #     assert n == m
    #     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    #
    # # Barlow Twins 是一种无监督学习技术，旨在通过最大化不同视图（通常是同一图像的变形或裁剪版本）的表示之间的相关性来学习高质量的视觉特征。
    # def BarlowTwins_loss_fun(self, x, y):
    #     z1 = self.projector(x)  # (49,768)->(49,768)
    #     z2 = self.projector(y)
    #
    #     # empirical cross-correlation matrix
    #     c = self.bn(z1).T @ self.bn(z2)
    #     c.div_(z1.shape[0])
    #
    #     # c = torch.mm(z1.T, z2) / z1.size(0)
    #
    #     # sum the cross-correlation matrix between all gpus
    #     torch.distributed.all_reduce(c)  # 跨GPU求和
    #     # 对角线元素（即各特征维度自身的方差）接近1，意味着每个维度都有足够的方差来携带信息，通过惩罚非对角线元素（即不同特征维度间的协方差）来减少不同特征维度之间的冗余
    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # 计算交叉相关矩阵的对角线元素和非对角线元素的损失
    #     off_diag = self.off_diagonal(c).pow_(2).sum()
    #     loss = on_diag + self.lambd * off_diag
    #     return loss  # 调整

    def recon_image(self, image_embed, image):
        # reshape the cross-modal features to the same shape as image (bs,49,768)->(bs,768,7,7)
        # self.text_to_local_image_embed_stacks = self.text_to_local_image_embed_stacks.view(
        #     -1, self.text_to_local_image_embed_stacks.size(-1), *self.image_encoder.get_last_spatial_info())
        # # reshape the image features to the [B, C, H, W] (bs,49,768)->(bs,768,7,7)
        # image_embed = image_embed.view(-1, image_embed.size(-1), *self.image_encoder.get_last_spatial_info())
        # 选择注释下面的代码
        output = self.image_decoder(image_embed, self.text_to_local_image_embed_stacks, image)  # resnet:(bs,49,768) vit: ()
        rec_image_loss = output['loss']
        return rec_image_loss

    # def padding_sentence_stacks(self, sentence_stacks, max_length=16):
    #     # padding sentence_stacks to the same length
    #     # sentence_stacks: list
    #     # length: int
    #     # return: [B, length, D] tensor, attention mask [B, length]
    #     batch_size = len(sentence_stacks)
    #     padded_sentence_stacks = torch.zeros(batch_size, max_length, self.embed_dim).to(
    #         sentence_stacks[0].device)  # (bs,16,768)
    #     # trancated_sentence_stacks = []
    #     attention_mask = torch.zeros(batch_size, max_length).to(sentence_stacks[0].device)  # (bs,16)
    #     for i, sentence_stack in enumerate(sentence_stacks):
    #         if len(sentence_stack) > max_length:  # 只取前max_length个句子进行填充，并相应位置的注意力掩码设为1
    #             padded_sentence_stacks[i, :] = sentence_stack[:max_length]
    #             attention_mask[i, :] = 1
    #             # trancated_sentence_stacks.append(sentence_stack[:max_length])
    #         else:  # 直接复制整个堆叠到对应位置，并根据实际长度设置注意力掩码。
    #             padded_sentence_stacks[i, :len(sentence_stack)] = sentence_stack
    #             attention_mask[i, :len(sentence_stack)] = 1
    #             # trancated_sentence_stacks.append(sentence_stack)
    #     return padded_sentence_stacks, attention_mask

    def get_global_text_representation(self, local_text_embed_stacks):
        batch_stacks = []
        for local_text_embed in local_text_embed_stacks:  # 遍历bs
            batch_stacks.append(
                self.global_text_attention(local_text_embed.unsqueeze(dim=0)))  # 取出每个句子计算全局文本注意力(sen_n,768)(1,768)
        return torch.cat(batch_stacks, dim=0)

    # def trancated_sentence_stack(self, stacks, max_length=16):
    #     # truncate sentence_stack to max_length 将句子堆栈截断为最大长度
    #     # stacks: list of [L, D]
    #     # return: [B, length, D]
    #     truncated_stacks = []
    #     for i, stack in enumerate(stacks):
    #         if len(stack) > max_length:
    #             truncated_stacks.append(stack[:max_length])
    #         else:
    #             truncated_stacks.append(stack)
    #     return truncated_stacks

    # def padding_embed_ind_stacks(self, embed_ind, max_length=16):
    #     batch_size = len(embed_ind)
    #     # TODO: change the -1 flag to padding token index, which may speed up the matching process
    #     padded_embed_ind = torch.ones(batch_size, max_length).to(embed_ind[0].device) * -1  # -1 flag for debugging
    #     for i, embed_ind_stack in enumerate(embed_ind):
    #         if len(embed_ind_stack) > max_length:
    #             padded_embed_ind[i, :] = embed_ind_stack[:max_length]
    #         else:
    #             padded_embed_ind[i, :len(embed_ind_stack)] = embed_ind_stack
    #     return padded_embed_ind

    # def sinkhorn(self, Q, nmb_iters):
    #     '''
    #         :param Q: (num_prototypes, batch size)
    #
    #     '''
    #     with torch.no_grad():
    #         sum_Q = torch.sum(Q)
    #         Q /= sum_Q
    #
    #         K, B = Q.shape
    #
    #         if self.gpus > 0:
    #             u = torch.zeros(K).cuda()
    #             r = torch.ones(K).cuda() / K
    #             c = torch.ones(B).cuda() / B
    #         else:
    #             u = torch.zeros(K)
    #             r = torch.ones(K) / K
    #             c = torch.ones(B) / B
    #
    #         for _ in range(nmb_iters):
    #             u = torch.sum(Q, dim=1)
    #             Q *= (r / u).unsqueeze(1)
    #             Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    #
    #         return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
    #
    # def distributed_sinkhorn(self, Q, nmb_iters):
    #     with torch.no_grad():
    #         sum_Q = torch.sum(Q)
    #         dist.all_reduce(sum_Q)
    #         Q /= sum_Q
    #
    #         if self.gpus > 0:
    #             u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
    #             r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
    #             c = torch.ones(Q.shape[1]).cuda(
    #                 non_blocking=True) / (self.gpus * Q.shape[1])
    #         else:
    #             u = torch.zeros(Q.shape[0])
    #             r = torch.ones(Q.shape[0]) / Q.shape[0]
    #             c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])
    #
    #         curr_sum = torch.sum(Q, dim=1)
    #         dist.all_reduce(curr_sum)
    #
    #         for it in range(nmb_iters):
    #             u = curr_sum
    #             Q *= (r / u).unsqueeze(1)
    #             Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
    #             curr_sum = torch.sum(Q, dim=1)
    #             dist.all_reduce(curr_sum)
    #         return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
    #
    # #### Cross-model prototype alignment ####
    # def cpa(self, global_image_embed, global_text_embed, flag):
    #     with torch.no_grad():  # 将一个模型层的权重参数进行归一化处理，并复制回原来的模型层
    #         w = self.prototype_layer.weight.data.clone()
    #         w = F.normalize(w, dim=1, p=2)  # [500,128]
    #         self.prototype_layer.weight.copy_(w)
    #
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
    #
    #     # 对分配码应用softmax函数，得到概率分布，就是软聚类分配码q
    #     img_proto_prob = F.softmax(
    #         img_proto_out / self.proto_temperature, dim=1)
    #     report_proto_prob = F.softmax(
    #         report_proto_out / self.proto_temperature, dim=1)
    #
    #     # 计算图像到报告的对比损失，用负对数似然函数
    #     loss_i2t_proto = - \
    #         torch.mean(
    #             torch.sum(img_code * torch.log(report_proto_prob),
    #                       dim=1))  # 衡量了图像的分配码与报告的原型概率分布之间的匹配程度，表示图像和报告在原型空间中的对应更加一致，即它们的信息内容更加相关。
    #     loss_t2i_proto = - \
    #         torch.mean(torch.sum(report_code *
    #                              torch.log(img_proto_prob), dim=1))
    #
    #     loss_proto = (loss_i2t_proto + loss_t2i_proto) / 2.  # L_CPA
    #     cap_loss = loss_proto
    #     return cap_loss

    # def generate_mask_matrix(self, matrix):  # 掩码句子
    #     batch_size, num_sentences, sentence_dim = matrix.shape
    #     device = matrix.device  # 假设matrix是一个张量，可以直接获取设备信息
    #     # 在与输入相同设备上创建全1矩阵
    #     mask_matrix = torch.ones((batch_size, num_sentences, sentence_dim), device=device)
    #     # 随机选择8个句子索引进行置零，同样确保在相同设备上
    #     indices_to_zero = torch.randperm(num_sentences, device=device)[:8]
    #     # 对选中的句子索引位置置零
    #     mask_matrix[:, indices_to_zero, :] = 0
    #     return mask_matrix

    # def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
    #     if masked_indices is None:  # 如果没有提供masked_indices，则使用torch.bernoulli函数根据probability_matrix生成一个布尔张量，指示哪些位置的token应该被掩盖。
    #         masked_indices = torch.bernoulli(probability_matrix).bool()
    #     # 确保[PAD]和[CLS]等特殊token不被掩盖，因为它们在预训练中通常有特定用途
    #     masked_indices[input_ids == self.tokenizer.pad_token_id] = False
    #     masked_indices[input_ids == self.tokenizer.cls_token_id] = False
    #
    #     if targets is not None:  # 如果提供了targets，则将未被选中进行掩盖的位置的目标值设为-100。这在后续计算损失时会忽略这些位置，因为-100在很多损失函数中被视为忽略值
    #         targets[~masked_indices] = -100  # We only compute loss on masked tokens
    #
    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    #     input_ids[indices_replaced] = self.tokenizer.mask_token_id
    #
    #     # 10% of the time, we replace masked input tokens with random word
    #     indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
    #     input_ids[indices_random] = random_words[indices_random]
    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #
    #     if targets is not None:
    #         return input_ids, targets
    #     else:
    #         return input_ids

    def stage1_step(self, batch, split="train"):
        image = batch['image']  # (bs,3,224,224)
        text = batch['text']  # input_id:(bs,256)
        '''
        =================================================================
        Encode image and text and get the local and global representation
        =================================================================
        '''
        # Embed image
        local_image_embed, global_image_embed = self.encode_image(image)  # (bs,49,768),(bs,768)
        # local_image_embed, global_image_embed = checkpoint(self.encode_image, image)
        local_image_embed = F.normalize(local_image_embed, dim=-1)
        global_image_embed = F.normalize(global_image_embed, dim=-1)
        # image_atts = torch.ones(local_image_embed.size()[:-1], dtype=torch.long).to(image.device)

        # Embed text
        local_text_embed, _ = self.encode_text(text)  # (bs,256)->(bs,256,768)
        # local_text_embed, _ = checkpoint(self.encode_text, text)
        # local_text_embed, _ = checkpoint(self.encode_text, text, preserve_rng_state=True, use_reentrant=False)
        local_text_embed = F.normalize(local_text_embed, dim=-1)
        # gather local text embedding on sentence level
        local_text_embed_stacks,_ = self.item_gather(local_text_embed, batch)  # batchsize,(句子个数,768)
        # get global text embedding  通过句子级注意力池化
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)  # (bs,768)
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
        # compute retrieval accuracy 即给定一个文本，找到最匹配的图像，或者给定一个图像，找到最匹配的文本
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            logits_per_image, labels, top_k=(1, 5))  # 获取分数最高的前k个索引,判断前k个索引中是否包含正确的标签
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            logits_per_text, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        '''
        =================================================================
        Calculate cap loss
        =================================================================
        '''
        # cpa_loss = self.cpa(global_image_embed, global_text_embed, flag='stage1')

        '''

        =================================================================
        Calculate the MLM loss
        =================================================================
        '''
        # input_ids = text.input_ids.clone()
        # labels = input_ids.clone()
        #
        # probability_matrix = torch.full(labels.shape, self.mlm_probability)  # (16,15)
        # input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
        #                               probability_matrix=probability_matrix)
        #
        # mlm_output = self.text_encoder(input_ids,
        #                                attention_mask=text.attention_mask,
        #                                encoder_hidden_states=local_image_embed,
        #                                encoder_attention_mask=image_atts,
        #                                return_dict=True,
        #                                labels=labels,
        #                                )
        # loss_mlm = mlm_output.loss
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
            loss = local_image_loss + local_text_loss + global_alignment_loss  # + cpa_loss
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
            loss = local_image_loss + local_text_loss + global_alignment_loss  # + cpa_loss
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
        local_text_embed, _ = self.encode_text(text)  # (bs,256)->(bs,256,768)
        # local_text_embed, _ = checkpoint(self.encode_text, text, preserve_rng_state=True, use_reentrant=False)
        local_text_embed = F.normalize(local_text_embed, dim=-1)
        # gather local text embedding on sentence level
        local_text_embed_stacks,_ = self.item_gather(local_text_embed, batch)  # bs,(sen_n,768)//每个sen_n可变
        # get global text embedding  通过句子级注意力池化
        global_text_embed = self.get_global_text_representation(local_text_embed_stacks)  # (bs,768)
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
        # compute retrieval accuracy 即给定一个文本，找到最匹配的图像，或者给定一个图像，找到最匹配的文本
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            logits_per_image, labels, top_k=(1, 5))  # 获取分数最高的前k个索引,判断前k个索引中是否包含正确的标签
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            logits_per_text, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        # local alignment loss: local_image_loss,local_text_loss
        local_image_loss, local_text_loss = self.local_alignment_loss(local_image_embed,
                                                                      local_text_embed_stacks)  # (bs,49,768)  bs,(sen_n,768)

        '''
        =================================================================
        Calculate cap loss
        =================================================================
        '''
        # cpa_loss = self.cpa(global_image_embed, global_text_embed, flag='stage2')
        '''

        =================================================================
        Calculate the MLM loss
        =================================================================
        '''
        # input_ids = text.input_ids.clone()
        # labels = input_ids.clone()
        #
        # probability_matrix = torch.full(labels.shape, self.mlm_probability)  # (16,15)
        # input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
        #                               probability_matrix=probability_matrix)
        #
        # mlm_output = self.text_encoder(input_ids,
        #                                attention_mask=text.attention_mask,
        #                                encoder_hidden_states=local_image_embed,
        #                                encoder_attention_mask=image_atts,
        #                                return_dict=True,
        #                                labels=labels,
        #                                )
        # loss_mlm = mlm_output.loss
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
            loss = local_image_loss + local_text_loss + global_alignment_loss + rec_image_loss  # +cpa_loss
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
            loss = local_image_loss + local_text_loss + global_alignment_loss + rec_image_loss # + cpa_loss
            loss_dict["val_stage2_loss"] = loss
            return local_image_embed, loss_dict

    def forward(self, batch):
        image_features = self.image_encoder(batch)
        text_features = self.text_encoder(batch)
        return image_features, text_features

    # def on_train_epoch_start(self) -> None:
    #     if self.current_epoch == 0:
    #         # Stage 1 starts
    #         optimizers, lr_schedulers, _ = self.call_optimization(max_epochs=self.stage1_epochs,
    #                                                               warmup_epochs=self.stage1_warmup_epochs,
    #                                                               weight_decay=self.stage1_weight_decay,
    #                                                               learning_rate=self.stage1_learning_rate,
    #                                                               learning_rate_start=self.stage1_learning_rate_start,
    #                                                               learning_rate_end=self.stage1_learning_rate_end)
    #         self.trainer.lr_schedulers = lr_schedulers
    #         self.trainer.optimizers = optimizers
    #     elif self.current_epoch == self.stage1_epochs:
    #         optimizers, lr_schedulers, _ = self.call_optimization(max_epochs=self.stage2_epochs,
    #                                                               warmup_epochs=self.stage2_warmup_epochs,
    #                                                               weight_decay=self.stage2_weight_decay,
    #                                                               learning_rate=self.stage2_learning_rate,
    #                                                               learning_rate_start=self.stage2_learning_rate_start,
    #                                                               learning_rate_end=self.stage2_learning_rate_end)
    #         self.trainer.lr_schedulers = lr_schedulers
    #         self.trainer.optimizers = optimizers

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

    # def on_train_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
    #     if self.current_epoch >= self.stage1_epochs:  # 条件决定了是否开始应用温度调整逻辑
    #         current_step = self.global_step - self.stage1_epochs * self.train_iters_per_epoch
    #         total_step = self.stage2_epochs * self.train_iters_per_epoch  #  总步数现在仅为第二阶段的迭代次数
    #         self.sentence_bank.set_temp(current_step, total_step, self.temp_decay)
    #     self.log('spb_temp', self.sentence_bank.curr_temp, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.stage1_epochs - 1:
            if self.global_rank == 0:
                self.trainer.save_checkpoint(f"{self.best_path}/stage1_end.ckpt")

        # 单独保存第二阶段权重
        if self.current_epoch > self.stage1_epochs - 1:
            # self.min_val_loss = float('inf')  # 重置损失开启二阶段
            current_val_loss = self.trainer.callback_metrics['val_loss'].item()
            # 检查当前 val_loss 是否小于已记录的最小 val_loss
            if current_val_loss < self.min_val_loss:
                self.min_val_loss = current_val_loss
                print(f"new_val_loss:{self.min_val_loss}")
                # 只有 global_rank 为0的进程保存模型
                if self.global_rank == 0:
                    # 保存当前最佳模型
                    self.trainer.save_checkpoint(f"{self.best_path}/best_model_stage2.ckpt")

        # ''' Save img_queue and report_queue for visualization '''
        # if self.local_rank == 0:
        #     img_queue_path = f"{self.trainer.callbacks[-1].dirpath}/img_queue.pth"
        #     torch.save(self.img_queue, img_queue_path)
        #     report_queue_path = f"{self.trainer.callbacks[-1].dirpath}/report_queue.pth"
        #     torch.save(self.report_queue, report_queue_path)

    def call_optimization(self, max_epochs=None, warmup_epochs=None, learning_rate=None, learning_rate_start=None,
                          learning_rate_end=None, weight_decay=None, slow_text_encoder=False):
        optim_conf = self.configure_optimizers(max_epochs=max_epochs, warmup_epochs=warmup_epochs,
                                               slow_text_encoder=slow_text_encoder, learning_rate=learning_rate,
                                               learning_rate_start=learning_rate_start,
                                               learning_rate_end=learning_rate_end, weight_decay=weight_decay)
        optimizers, lr_schedulers, optimizer_frequencies, monitor = self.trainer._configure_optimizers(optim_conf)
        lr_schedulers = self.trainer._configure_schedulers(lr_schedulers, monitor, not self.automatic_optimization)
        return optimizers, lr_schedulers, optimizer_frequencies

    # forward关注于模型的输入到输出映射，而training_step则专注于训练过程的实现，包括损失计算和日志记录
    def training_step(self, batch,
                      batch_idx):  # 使用PyTorch-Lightning时，在trining_step定义损失，可以return一个tensor会作为Loss回传，也可以return一个字典
        if self.current_epoch < self.stage1_epochs:
            image_feaures, loss_dict = self.stage1_step(batch)
            loss = loss_dict['stage1_loss']
        elif self.current_epoch >= self.stage1_epochs:
            image_feaures, loss_dict = self.stage2_step(batch)
            loss = loss_dict['stage2_loss']
        self.log_dict(loss_dict, on_step=True, on_epoch=True, sync_dist=True,
                      prog_bar=False)  # log_dict 是一种便捷的方式来同时记录多个指标或损失到日志中
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
                      prog_bar=True)  # log_dict 是一种便捷的方式来同时记录多个指标或损失到日志中
        # print("val_step_loss:{}".format(val_loss))

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

    # 从模型的参数中排除特定部分（在这里是"text_encoder"相关参数），以便对其进行不同的处理
    def exclude_from_text_encoder(self, named_params, weight_decay):
        # exclude discriminator param
        params = []
        excluded_params = []
        for name, param in named_params:
            if not param.requires_grad:  # 如果参数不需要梯度计算
                continue
            elif 'text_encoder' in name:
                excluded_params.append(param)
            else:
                params.append(param)
        return params, excluded_params  # 返回两组参数列表，一组是常规训练参数(params)，另一组是被排除的、与"text_encoder"相关的参数(excluded_params)

    # 为深度学习模型提供了灵活且精细的优化器与学习率调度配置能力，特别是针对含有特定模块（如"text_encoder"）的模型，可以实现差异化训练策略。

    def configure_optimizers(self, learning_rate=1e-5, learning_rate_start=1e-7, learning_rate_end=0, max_epochs=40,
                             warmup_epochs=1, slow_text_encoder=False, weight_decay=1e-6):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=weight_decay)
        else:
            params = self.parameters()
        if slow_text_encoder:  # 区分出与"text_encoder"相关的参数，并为这两组参数创建独立的参数组，可能为了对"text_encoder"使用不同的学习率。
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
            optimizer.param_groups[0]['lr'] = learning_rate / 10  # slow down text encoder
            optimizer.param_groups[1]['lr'] = learning_rate
        warmup_steps = self.train_iters_per_epoch * warmup_epochs
        total_steps = self.train_iters_per_epoch * max_epochs
        if self.scheduler == 'cosine_warmup_linear_annealing':
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",  # 学习率都会在每次前向传播和反向传播后更新权重的步骤进行调整，而不是等到整个 epoch 完成后再调整。
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

    # freeze prototype layer  # 先让模型的其他部分（如特征提取层）有足够的时间学习到较好的特征表示，然后再解冻原型层，让原型根据这些特征进一步优化，以更好地捕捉数据中的结构和模式
    # def on_after_backward(self):
    #     if self.current_epoch < self.hparams.freeze_prototypes_epochs:
    #         for param in self.prototype_layer.parameters():
    #             param.grad = None

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
        parser.add_argument("--img_encoder", type=str, default="resnet_50")  # biformer_tiny,resnet_50,vit_base
        parser.add_argument("--freeze_bert", default=False, action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=768, help="768")
        parser.add_argument("--num_workers", type=int, default=20)

        parser.add_argument("--batch_size", type=int, default=64)  # 64
        parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                            help='weight on off-diagonal terms')
        parser.add_argument("--stage1_epochs", type=int, default=10)  # 10
        parser.add_argument("--stage2_epochs", type=int, default=20)  # 20
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
        dataset_size = len(dataset)  # # 计算训练数据集的大小
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)  # 计算可用的设备数量
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        # 计算有效的批次大小，等于训练器对象的accumulate_grad_batches乘以设备数量
        # 计算总的训练步数，等于数据集大小除以有效批次大小，再乘以训练器对象的max_epochs
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
    parser = ArgumentParser()  # 创建一个ArgumentParser对象，它可以用来解析命令行参数。ArgumentParser对象可以指定程序需要哪些参数，以及如何从sys.argv中获取它们。
    # trainer args (gpus,max_epochs,gradient_clip_val,precision,log_every_n_steps,fast_dev_run)
    parser = Trainer.add_argparse_args(parser)  # 将Trainer类的默认参数添加到一个ArgumentParser对象中，这样可以方便地从命令行接收Trainer的参数
    # model args
    parser = CENSJL.add_model_specific_args(parser)  # 将MGCA模型的特定参数添加到一个ArgumentParser对象中，这样可以方便地从命令行接收MGCA的参数
    args = parser.parse_args()
    args.deterministic = True  # 是否使用“确定性”算法，也就是说，给定相同的输入，在相同的软件和硬件上，总是产生相同的输出。
    seed_everything(args.seed)  # 固定所有随机数生成器的种子
    args.max_epochs = args.stage1_epochs + args.stage2_epochs
    args.precision = 16
    # 调试
    # args.limit_train_batches = 0.1
    # args.limit_val_batches = 0.1
    args.num_sanity_val_steps = 0  # 开始训练之前是否进行两批验证

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    args.best_path = ckpt_dir

    # Add load from checkpoint
    model = CENSJL(**args.__dict__)  # 根据 args 对象的字典表示来创建 MGCA 类的一个实例.**操作符进行“关键字参数解包”,每个键值对都会作为单独的参数传入到MGCA类的构造函数中

    # 指定要从中恢复的检查点的路径
    should_resume = False
    if should_resume:
        args.resume_from_checkpoint = "/home/lihongxing/project/MGCA/data/ckpts/MGCA/2024_08_03_14_18_36/last.ckpt"
        model.kc.initialized = True

    datamodule = DataModule(MultimodalPretrainingDataset, custom_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers)

    # datamodule = DataModule_pretrain(MultimodalPretrainingDataset, custom_collate_fn,
    #                         DataTransforms, args.data_pct,
    #                         args.batch_size, args.num_workers)  # DataModule_pretrain
    # datamodule.prepare_data_h5()
    # return

    callbacks = [
        LearningRateMonitor(logging_interval="step"),  # 是一个用于监控和记录学习率的回调函数，它会在每个训练步骤（即每个批次结束）时记录所有优化器的学习率
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1)
        # EarlyStopping(monitor="val_loss", min_delta=0.00001,
        #               patience=10, verbose=True, mode="min")
    ]

    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")  # 记录数据的地址
    # os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="CENS", save_dir=logger_dir, name=extension)  # 在wandb网页上面可视化结果
    # 使用PyTorch Lightning库来创建一个Trainer对象，它可以用来训练和验证一个PyTorch模型。Trainer对象的参数是从argparse模块解析得到的
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,  # 包含回调函数的列表
        logger=wandb_logger)  # 让pytorch lightning不自动创建checkpoint文件夹

    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
