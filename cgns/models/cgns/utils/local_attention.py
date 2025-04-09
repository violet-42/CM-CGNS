import torch.nn as nn
import torch
import math
# class LocalCrossAttention2(nn.Module):
#     def __init__(self, embed_dim, drop_rate=0):
#         super(LocalCrossAttention2, self).__init__()
#         self.embed_dim = embed_dim
#
#         # 添加LayerNorm层用于前/后归一化
#         self.norm_img = nn.LayerNorm(embed_dim)
#         self.norm_txt = nn.LayerNorm(embed_dim)
#
#         self.query1 = nn.Linear(embed_dim, embed_dim)
#         self.key1 = nn.Linear(embed_dim, embed_dim)
#         self.value1 = nn.Linear(embed_dim, embed_dim)
#         self.dropout1 = nn.Dropout(drop_rate)
#         self.query2 = nn.Linear(embed_dim, embed_dim)
#         self.key2 = nn.Linear(embed_dim, embed_dim)
#         self.value2 = nn.Linear(embed_dim, embed_dim)
#         self.dropout2 = nn.Dropout(drop_rate)
#
#     def forward(
#             self,
#             input_tensor1,
#             input_tensor2,
#             attention_mask1=None,
#             attention_mask2=None
#     ):
#         # for vision input [I_N, D]
#         query_layer1 = self.query1(input_tensor1)  # (49,768)
#         key_layer1 = self.key1(input_tensor1)  # (49,768)
#         value_layer1 = self.value1(input_tensor1)  # (49,768)
#
#         # for text input [T_N,D]
#         query_layer2 = self.query2(input_tensor2)  # (sen_n,768)
#         key_layer2 = self.key2(input_tensor2)
#         value_layer2 = self.value2(input_tensor2)
#
#         attention_scores1 = torch.bmm(query_layer2, key_layer1.permute(0, 2, 1))  # [T_N, D] @ [D, I_N] = [T_N, I_N] (sen_n,49)
#         attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)  # (15,49)
#         if attention_mask1 is not None:
#             attention_scores1 = attention_scores1 + attention_mask1
#         # Sigmoid is better in this case
#         # TODO: pre-normalize vs. post-normalize
#         attention_probs1 = torch.sigmoid(attention_scores1)
#         attention_probs1 = self.dropout1(attention_probs1)
#         context_layer1 = torch.bmm(attention_probs1,value_layer1)  # (bs,16,768) image_to_local_text的跨模态文本表征
#
#         attention_scores2 = torch.bmm(query_layer1,key_layer2.permute(0, 2, 1))  # [I_N, D] @ [D, T_N] = [I_N, T_N]
#         attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)
#         if attention_mask2 is not None:
#             attention_scores2 = attention_scores2 + attention_mask2
#         attention_probs2 = torch.sigmoid(attention_scores2)
#         attention_probs2 = self.dropout2(attention_probs2)
#         context_layer2 = torch.bmm(attention_probs2, value_layer2)  # (bs,49,768) text_to_local_image的跨模态图片表征
#
#         # 后归一化
#         context_layer1 = self.norm_img(context_layer1)
#         context_layer2 = self.norm_txt(context_layer2)
#         return context_layer2, attention_probs2, context_layer1, attention_probs1

class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
            self,
            input_tensor1,
            input_tensor2,
            attention_mask1=None,
            attention_mask2=None
    ):
        # for vision input [I_N, D]
        query_layer1 = self.query1(input_tensor1)  # (49,768)
        key_layer1 = self.key1(input_tensor1)  # (49,768)
        value_layer1 = self.value1(input_tensor1)  # (49,768)

        # for text input [T_N,D]
        query_layer2 = self.query2(input_tensor2)  # (sen_n,768)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 = self.value2(input_tensor2)

        attention_scores1 = query_layer2 @ key_layer1.T  # [T_N, D] @ [D, I_N] = [T_N, I_N] (sen_n,49)
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)  # (15,49)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1
        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize
        attention_probs1 = torch.sigmoid(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = attention_probs1 @ value_layer1  # [T_N, I_N] @ [I_N,D] = [T_N, D] image_to_local_text的跨模态文本表征

        attention_scores2 = query_layer1 @ key_layer2.T  # [I_N, D] @ [D, T_N] = [I_N, T_N]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)
        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2
        attention_probs2 = torch.sigmoid(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = attention_probs2 @ value_layer2  # [I_N, T_N] @ [T_N, D] = [I_N, D] text_to_local_image的跨模态图片表征
        return context_layer2, attention_probs2, context_layer1, attention_probs1