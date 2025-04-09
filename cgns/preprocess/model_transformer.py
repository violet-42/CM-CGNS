import torch
import os
import sys
import torch.nn as nn
from torchvision.models import resnet50
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('/home1/lihongxing/project/lovt-main/src/models')

def add_prefix_to_state_dict(state_dict, prefix):
    ''' Add a prefix to all keys in the state dict '''
    return {prefix + key: value for key, value in state_dict.items()}
def load_encoder_from_ckpt(ckpt_path, output_dir, model_name):
    # checkpoint = torch.load(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    print(checkpoint.keys())

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint 

    model = resnet50()

    if model_name=='mgca':
        filtered_state_dict = {}
        # 遍历状态字典的每一项
        for key, value in state_dict.items():
            # 检查键是否以指定的前缀开始
            if key.startswith('img_encoder_q.model.'):
                # 移除前缀并存储到新字典中
                new_key = key.replace('img_encoder_q.model.', '')
                filtered_state_dict[new_key] = value
        # state_dict_with_prefix = add_prefix_to_state_dict(filtered_state_dict, 'encoder.')

    elif model_name=='ours':
        filtered_state_dict = {}
        # 遍历状态字典的每一项
        for key, value in state_dict.items():
            # 检查键是否以指定的前缀开始
            if key.startswith('image_encoder.model.'):
                # 移除前缀并存储到新字典中
                new_key = key.replace('image_encoder.model.', '')
                filtered_state_dict[new_key] = value

    elif model_name=='prior':
        # if list(state_dict['conv1.weight'].size())[1] != 3:
        #     # Duplicate the single channel weights to all three channels
        #     state_dict['conv1.weight'] = state_dict['conv1.weight'].repeat(1, 3, 1, 1) / 3  # 将预训练权重复制到所有三个输入通道上。这通常适用于灰度图像预训练模型转换到RGB图像
        #     filtered_state_dict = state_dict
        filtered_state_dict = {}
        # 遍历状态字典的每一项
        for key, value in state_dict.items():
            # 检查键是否以指定的前缀开始
            if key.startswith('image_encoder.encoder.'):
                # 移除前缀并存储到新字典中
                new_key = key.replace('image_encoder.encoder.', '')
                filtered_state_dict[new_key] = value
        if list(filtered_state_dict['conv1.weight'].size())[1] != 3:
            # Duplicate the single channel weights to all three channels
            filtered_state_dict['conv1.weight'] = filtered_state_dict['conv1.weight'].repeat(1, 3, 1, 1) / 3  # 将预训练权重复制到所有三个输入通道上。这通常适用于灰度图像预训练模型转换到RGB图像
            filtered_state_dict = filtered_state_dict
        # state_dict_with_prefix = add_prefix_to_state_dict(filtered_state_dict, 'encoder.')

    elif model_name=='biovil':
        filtered_state_dict = {}
        # 遍历状态字典的每一项
        for key, value in state_dict.items():
            # 检查键是否以指定的前缀开始
            if key.startswith('encoder.encoder.'):
                # 移除前缀并存储到新字典中
                new_key = key.replace('encoder.encoder.', '')
                filtered_state_dict[new_key] = value


    print(model.state_dict().keys())
    print(filtered_state_dict.keys())

    model.load_state_dict(filtered_state_dict, strict=False)

    # 保存编码器
    torch.save(model.state_dict(), f"{output_dir}/CENS-JL22.pth")


# 假设你有多个检查点文件
checkpoint_files = [
    "/home1/lihongxing/temp/CENS-JL22.ckpt",
    # ...
]

output_directory = "/home1/lihongxing/checkpoints/"

for ckpt_file in checkpoint_files:
    load_encoder_from_ckpt(ckpt_file, output_directory, model_name='ours')  # mgca ours prior

print("All encoders have been extracted and saved.")

''''''
# import torch
# from timm import create_model
# from timm.models.vision_transformer import checkpoint_filter_fn
# from pathlib import Path
# from mgca.models.backbones.vits import create_vit
#
# # 定义模型保存的位置
# checkpoint_path = '/home1/lihongxing/temp/mgca_vit_base.ckpt'
# checkpoint_path = Path(checkpoint_path)
#
# # 创建 Vision Transformer 基础模型实例
# vit_name = 'base'
# image_size = 224
# vit_grad_ckpt = False  # 是否使用梯度检查点
# vit_ckpt_layer = 0  # 如果使用梯度检查点，则指定从哪一层开始使用。
# model, vision_width = create_vit(
#                 vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)
# print(model.state_dict().keys())
#
# # 加载模型权重
# if checkpoint_path.exists():
#     # 使用 GPU 加载检查点
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#
#     # 获取模型权重
#     state_dict = checkpoint.get('state_dict', checkpoint)
#
#     # 定义一个函数来过滤和重命名键
#     def rename_keys(state_dict):
#         new_state_dict = {}
#         for k, v in state_dict.items():
#             # 移除前缀 "img_encoder_q.model."
#             if k.startswith('img_encoder_q.model.'):  # mgca: img_encoder_q.model.  censjl: image_encoder.model.
#                 new_k = k[len('img_encoder_q.model.'):]
#                 new_state_dict[new_k] = v
#             else:
#                 # new_state_dict[k] = v
#                 continue
#         return new_state_dict
#
#
#     state_dict = rename_keys(state_dict)
#     state_dict = checkpoint_filter_fn(state_dict, model)  # 调整状态字典以匹配模型
#     print(state_dict.keys())
#
#     # 加载调整后的状态字典
#     model.load_state_dict(state_dict)
# else:
#     print("Checkpoint not found")
# output_directory = "/home1/lihongxing/checkpoints/"
# torch.save(model.state_dict(), f"{output_directory}/mgca_vit.pth")