import os

import torch
import torch.nn as nn
from einops import rearrange
from cgns.models.backbones import cnn_backbones
from cgns.models.backbones.med import BertModel
from cgns.models.backbones.vits import create_vit
from transformers import AutoTokenizer, BertConfig, BertTokenizer, logging,AutoModel,BertModel
from cgns.models.backbones.encoder_utils.biformer import BiFormerModel

import torch.nn.functional as F

logging.set_verbosity_error()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]  # (50,64,2048)
class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [72, 768, 196]
        x = self.head(x)  # [72, 128, 196]

        return x.permute(0, 2, 1)


class ImageEncoder(nn.Module):
    def __init__(self,
                 model_name: str = "resnet_50",
                 pretrained: bool = True
                 ):
        super(ImageEncoder, self).__init__()

        self.model_name = model_name  # vit_base

        if model_name == 'vit_base':   # vit_base
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224
            #
            vit_name = model_name[4:]  # base
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
            # 梯度检查点和激活点检查点是两种节约显存的方法。
            self.feature_dim = vision_width  # 768/1024
            self.patch_embedding = self.model.patch_embed

            # checkpoint = torch.hub.load_state_dict_from_url(
            #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            #     map_location="cpu", check_hash=True)  # 下载模型的状态字典

            checkpoint = torch.load("/home/***/project/CGNS/run_file/deit_base_patch16_224-b5f2ef4d.pth")  # imagenet-1k
            state_dict = checkpoint["model"]
            # msg = self.model.load_state_dict(state_dict, strict=False)  # strict=False 表示当前的模型和预训练的模型不完全相同，所以只会加载两者共有的层的参数，而忽略其他的层
            self.model.load_state_dict(state_dict, strict=False)

        elif model_name == 'resnet_50':
            model_function = getattr(
                cnn_backbones, model_name)  # 根据model_name,从cnn_backbone模块中取出的函数对象
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=pretrained
            )

            # Average pooling
            # self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.pool = AttentionPool2d(spacial_dim=7, embed_dim=2048, num_heads=8)

        elif model_name == 'biformer_tiny':
            self.model = BiFormerModel(name=model_name,pretrained=True)
            self.feature_dim = self.model.feature_dim  # 编码器输出维度特征，在微调时会使用。

    def resnet_forward(self, x, get_local=True):
        # x = nn.Upsample(size=(299, 299), mode="bilinear",
        #                 align_corners=True)(x)
        x = self.model.conv1(x)  # (batch_size, 64, 112, 112)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # (bs,64,56,56)

        x = self.model.layer1(x)  # (bs, 256, 56, 56)  # 残差块
        x = self.model.layer2(x)  # (bs,512,28,28)
        x = self.model.layer3(x)  # (bs,1024,14,14)

        x = self.model.layer4(x)  # (bs,2048,7,7)
        local_features = x

        global_x = self.pool(x)  # AdaptiveAvgPool2d (bs,2048)
        global_x = global_x.view(global_x.size(0), -1)
        local_features = local_features.view(local_features.shape[0], -1, local_features.shape[1])  # (bs,49,2048)

        return global_x,local_features

    def vit_forward(self, x):
        return self.model(x)

    def biformer_forward(self, x):
        return self.model(x)

    def forward(self, x, get_local=False):
        if "resnet" in self.model_name:
            return self.resnet_forward(x, get_local=get_local)
        elif "vit" in self.model_name:
            img_feat = self.vit_forward(x)  # [bs, 197, 768]
            return img_feat[:, 0].contiguous(), img_feat[:, 1:].contiguous()  # 第一个是cls token
        elif "biformer" in self.model_name:
            return self.biformer_forward(x)
        else:
            raise NotImplementedError

    def get_last_spatial_info(self):
        if self.model_name == 'resnet_50':
            return [7, 7]
        elif self.model_name == 'vit_base':
            return [7, 7]
        elif self.model_name == 'biformer_tiny':
            return [7, 7]
        else:
            raise ValueError(f'{self.model_name}is not surpported!')

    def get_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4

    def get_global_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4

    def get_local_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4



class BertEncoder(nn.Module):
    def __init__(self, tokenizer: BertTokenizer = None, pretrained=True, vocab_size=28996):
        super().__init__()
        self.agg_tokens = True
        self.last_n_layers = 1
        self.bert_type = "/home/***/project/CGNS/Bio_ClinicalBERT/"
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                "/home/***/project/CGNS/Bio_ClinicalBERT/")
        else:
            self.encoder = BertModel(BertConfig(hidden_size=768, vocab_size=vocab_size))
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)

        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def aggregate_tokens(self, embeddings, caption_ids, last_layer_attn):
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3) 
        agg_embs_batch = [] 
        sentences = [] 
        last_attns = [] 

        for embs, caption_id, last_attn in zip(embeddings, caption_ids, last_layer_attn):
            agg_embs = [] 
            token_bank = [] 
            words = [] 
            word_bank = [] 
            attns = [] 
            attn_bank = [] 

            for word_emb, word_id, attn in zip(embs, caption_id, last_attn):
                word = self.idxtoword[word_id.item()] 
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank) 
                    new_emb = new_emb.sum(axis=0) 

                    agg_embs.append(new_emb)
                    words.append("".join(word_bank)) 
                    attns.append(sum(attn_bank)) 

                    agg_embs.append(word_emb)
                    words.append(word)
                    attns.append(attn)
                    break
                if not word.startswith("##"): 
                    if len(word_bank) == 0: 
                        token_bank.append(word_emb)
                        word_bank.append(word)
                        attn_bank.append(attn)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)

                        agg_embs.append(new_emb)
                        words.append("".join(word_bank)) 
                        attns.append(sum(attn_bank))  
                        token_bank = [word_emb]
                        word_bank = [word]
                        attn_bank = [attn]
                else:  
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
                    attn_bank.append(attn)

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs) 
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.type_as(agg_embs)
            words = words + ["[PAD]"] * padding_size  
            last_attns.append(
                torch.cat([torch.tensor(attns), torch.zeros(padding_size)], dim=0))  
            agg_embs_batch.append(torch.cat([agg_embs, paddings])) 
            sentences.append(words)  

        agg_embs_batch = torch.stack(agg_embs_batch)  
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)  
        last_atten_pt = torch.stack(last_attns)  
        last_atten_pt = last_atten_pt.type_as(agg_embs_batch)

        return agg_embs_batch, sentences, last_atten_pt 

    def forward(self, x, return_dict=None,output_attentions=None):
        if output_attentions == None:
            return self.encoder(**x, return_dict=return_dict)  

        outputs = self.encoder(**x, return_dict=return_dict,output_attentions=output_attentions)  
        last_layer_attn = outputs.attentions[-1][:, :, 0, 1:].mean(dim=1) 
        all_feat = outputs.last_hidden_state.unsqueeze(
            1) 

        if self.agg_tokens:
            all_feat, sents, last_atten_pt = self.aggregate_tokens(
                all_feat, x['input_ids'], last_layer_attn)  
            last_atten_pt = last_atten_pt[:, 1:].contiguous()
        else:
            sents = [[self.idxtoword[w.item()] for w in sent]
                     for sent in x['input_ids']] 

        if self.last_n_layers == 1:
            all_feat = all_feat[:, 0] 

        report_feat = all_feat[:, 0].contiguous()  
        word_feat = all_feat[:, 1:].contiguous() 

        return report_feat, word_feat, last_atten_pt, sents  
        # return self.encoder(**x, return_dict=return_dict) 

    def get_width(self):
        # return self.encoder.hidden_siz
        return 768


if __name__ == "__main__":
    from cgns.datasets.pretrain_dataset import MultimodalPretrainingDataset
    from cgns.datasets.transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)

    for i, data in enumerate(dataset):
        imgs, caps, cap_len, key = data
        if caps["attention_mask"].sum() == 112:
            model = BertEncoder()
            report_feat, sent_feat, sent_mask, sents = model(
                caps["input_ids"],
                caps["attention_mask"],
                caps["token_type_ids"],
                get_local=True)
