import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch.base as md
from kornia.losses import ssim
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from timm.models.vision_transformer import PatchEmbed, Block
# from util.pos_embed import get_2d_sincos_pos_embed

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            output_size,
            connection_dropout=0.0,
            use_batchnorm=True,
            attention_type=None
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )  # 通道数减半
        self.attention1 = md.Attention(attention_type, in_channels=in_channels)  # Identity, SCSEModule
        self.output_size = output_size  # 14
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.connection_dropout_rate = connection_dropout
        self.dropout = nn.Dropout2d(connection_dropout)

    def forward(self, x, skip=None):
        if skip is not None:
            # print(x.shape, skip.shape)
            if self.connection_dropout_rate > 0:
                skip = self.dropout(skip)
            x = torch.cat([x, skip], dim=1)
        x = F.interpolate(x, self.output_size, mode="nearest") 
        x = self.conv1(x)  
        x = self.conv2(x)
        x = self.attention2(x)  # Identity
        return x

def get_features_size(encoder_name):
    if encoder_name == 'resnet_50':
        return [14, 28, 56, 112, 224], [2048, 1024, 512, 256, 64]  
    elif encoder_name == 'vit_base':  
        return 14  
    else:
        raise NotImplementedError

def clear(self, *args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    def _clear(f):
        if not hasattr(self, f):
            return
        attr = getattr(self, f)
        if isinstance(attr, torch.Tensor):
            attr.set_()
        elif isinstance(attr, list):
            del attr[:]
        else:
            setattr(self, f, None)
    for key in args:
        _clear(key)
    return self

class SpatialDropout(nn.Module):
    def __init__(self, p=0.5, is_train=True, encoder_name='resnet_50'):
        super(SpatialDropout, self).__init__()
        assert 0 <= p < 1, f"dropout probability has to be between 0 and 1, but got {p}"
        self.p = p
        self.is_train = is_train
        self.encoder_name = encoder_name

    def forward(self, input):
        if not self.is_train or self.p <= 0:
            return input

        if self.encoder_name == 'resnet_50':
            if input.dim() != 4:
                raise RuntimeError('Input must be 4D (nbatch, nfeat, h, w)')
            noise = input.new(input.size(0), input.size(1), 1, 1).bernoulli_(1 - self.p)
            return input * noise.expand_as(input)

        elif self.encoder_name == 'vit_base':
            if input.dim() != 3:
                raise RuntimeError('Input must be 3D (nbatch, nfeat, depth)')
            noise = input.new(input.size(0), input.size(1), 1).bernoulli_(1 - self.p)
            noise = noise / (1 - self.p)  
            return input * noise.expand_as(input)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p:.3f})"

class ImageDecoder_ori(nn.Module):
    def __init__(self, embed_dim, encoder_name='resnet_50', num_colors=3, image_dropout=0.5, dropout_mode='spatial', **kwargs) -> None:
        super().__init__()
        in_channels = []
        out_channels = []
        out_size = []
        size_list, channel_list = get_features_size(encoder_name)  # [14, 28, 56, 112, 224] [2048, 1024, 512, 256, 64]
        n_blocks = len(channel_list)  # 5
        in_ch = embed_dim*2  # 768*2
        if dropout_mode == 'channel':
            self.image_dropout = nn.Dropout2d(image_dropout)  # https://blog.csdn.net/qq_43391414/article/details/127227029
        elif dropout_mode == 'spatial':
            self.image_dropout = SpatialDropout(image_dropout, encoder_name) 
        for i in range(n_blocks):  
            in_channels.append(in_ch)  # [1536, 768, 384, 192, 96]
            out_channels.append(in_ch // 2)  # [768, 384, 192, 96, 48]
            out_size.append(size_list[i])  # [14, 28, 56, 112, 224]
            in_ch //= 2
        self.output_head = nn.Sequential(nn.Conv2d(in_ch, num_colors, kernel_size=3, padding=1)) 
        blocks = [
            DecoderBlock(in_ch, out_ch, size, **kwargs)
            for in_ch, out_ch, size in zip(in_channels, out_channels, out_size)
        ]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, image_embed, text_embed, image):  # resnet:(bs,768,7,7) ()
        # mask image features
        image_embed = image_embed.permute(0, 2, 1).contiguous()  # (bs,768,196) (bs,768,49)
        image_embed = image_embed.view(image_embed.size(0), image_embed.size(1), 7, 7)  # (bs,768,7,7)
        text_embed = text_embed.permute(0, 2, 1).contiguous()  # (bs,768,49)
        text_embed = text_embed.view(text_embed.size(0), text_embed.size(1), 7, 7)  # (bs,768,7,7)

        image_embed = self.image_dropout(image_embed) 
        # fuse image and text features
        z = torch.cat([image_embed, text_embed], dim=1)   # (4,768,7,7)->(4,1536,7,7)
        for i, block in enumerate(self.blocks): 
            z = block(z)
        output = self.output_head(z) 
        loss = self.loss_fn(output, image)
        return {'loss': loss, 'output': output}

    def loss_fn(self, recon_image, image):
        if image.size(2) != recon_image.size(2) or image.size(3) != recon_image.size(3):
            image = F.interpolate(image, size=recon_image.size()[2:], mode='bilinear', align_corners=True)
        recon_loss = F.l1_loss(recon_image, image)
        return recon_loss * 10

#######vit_decoder#######
class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):

        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        fused_data, _ = self.multihead_attn(query, key, value)
        fused_data = fused_data.transpose(0, 1)  # 再次转置回来
        return fused_data

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) 
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout) 
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        return output


def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])


class ImageDecoder_vit(nn.Module):
    def __init__(self, embed_dim, output_size=224, encoder_name='resnet_50', num_colors=3, image_dropout=0.5, dropout_mode='spatial', num_heads=8, num_layers=6, dim_feedforward=2048, activation="relu"):
        super().__init__()
        self.encoder_name = encoder_name

        if self.encoder_name == 'resnet_50':
            patch_num = 49
            self.output_head = nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 512, 14, 14)

                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 256, 28, 28)

                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 128, 56, 56)

                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 64, 112, 112)

                nn.Conv2d(64, num_colors, kernel_size=3, padding=1),  # 输出: (bs, 3, 224, 224)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Tanh()  
            )
        elif self.encoder_name == 'vit_base':
            patch_num = 196
            self.output_head = nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 512, 28, 28)

                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 256, 56, 56)

                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 128, 112, 112)

                nn.Conv2d(128, num_colors, kernel_size=3, padding=1),  # 输出: (bs, 3, 224, 224)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Tanh()
            )

        # Dropout
        if dropout_mode == 'channel': 
            self.image_dropout = nn.Dropout2d(image_dropout)  
        elif dropout_mode == 'spatial':
            self.image_dropout = SpatialDropout(image_dropout)
        self.output_size = output_size
        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, activation=activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers) 

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(patch_num, embed_dim))

        # Attention-based Fusion
        self.fusion_attention = MultiHeadAttentionFusion(embed_dim, num_heads)

    def forward(self, image_embed, text_embed, image):
        # Mask image features
        image_embed = self.image_dropout(image_embed) 

        # Fuse image and text features
        z = self.fusion_attention(query=image_embed, key=text_embed, value=text_embed)

        # Add positional encoding
        z = z + self.positional_encoding.unsqueeze(0)

        # Transformer decoding
        z = self.transformer_decoder(z, z)  

        # Reshape before feeding into ConvTranspose layers
        z = z.permute(0, 2, 1).contiguous()  # (bs,768,196) (bs,768,49)
        if self.encoder_name=='resnet_50':
            z = z.view(z.size(0), z.size(1), 7, 7)  # (bs,768,7,7)
        elif self.encoder_name=='vit_base':
            z = z.view(z.size(0), z.size(1), 14, 14)  # (bs,768,14,14)

        # Generate image from the reshaped feature map
        output = self.output_head(z)
        output = output.view(z.size(0), -1, self.output_size, self.output_size)
        loss = self.loss_fn(output, image)
        return {'loss': loss, 'output': output}

    def loss_fn(self, recon_image, image):
        if image.size(2) != recon_image.size(2) or image.size(3) != recon_image.size(3):
            image = F.interpolate(image, size=recon_image.size()[2:], mode='bilinear', align_corners=True)
        recon_loss = F.l1_loss(recon_image, image)
        return recon_loss * 10

class ImageDecoder(nn.Module):  
    def __init__(self, embed_dim, output_size=224, encoder_name='resnet_50', num_colors=3, image_dropout=0.5, dropout_mode='spatial', num_heads=8, num_layers=6, dim_feedforward=2048, activation="relu"):
        super().__init__()
        self.encoder_name = encoder_name
        self.dropout_mode = dropout_mode

        if self.encoder_name == 'resnet_50':
            patch_num = 49
            self.output_head = nn.Sequential(
                nn.Conv2d(embed_dim*2, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 512, 14, 14)

                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 256, 28, 28)

                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 128, 56, 56)

                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 64, 112, 112)

                nn.Conv2d(64, num_colors, kernel_size=3, padding=1),  # 输出: (bs, 3, 224, 224)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Tanh()  
            )
        elif self.encoder_name == 'vit_base':
            patch_num = 196
            self.output_head = nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 512, 28, 28)

                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 256, 56, 56)

                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 输出: (bs, 128, 112, 112)

                nn.Conv2d(128, num_colors, kernel_size=3, padding=1),  # 输出: (bs, 3, 224, 224)
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Tanh()  
            )

        # Dropout
        if self.dropout_mode == 'channel':  
            self.image_dropout = nn.Dropout2d(image_dropout)  
        elif self.dropout_mode == 'spatial':
            self.image_dropout = SpatialDropout(image_dropout,encoder_name=self.encoder_name)
        self.output_size = output_size
        # Transformer Decoder
        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, activation=activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)  

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(patch_num, embed_dim))

        # Attention-based Fusion
        self.fusion_attention = MultiHeadAttentionFusion(embed_dim, num_heads)

    def forward(self, image_embed, text_embed, image):
        # Reshape before fusion
        if self.encoder_name == 'resnet_50':
            image_embed = image_embed.permute(0, 2, 1).contiguous()  # (bs,768,196) (bs,768,49)
            image_embed = image_embed.view(image_embed.size(0), image_embed.size(1), 7, 7)  # (bs,768,7,7)
            text_embed = text_embed.permute(0, 2, 1).contiguous()  # (bs,768,49)
            text_embed = text_embed.view(text_embed.size(0), text_embed.size(1), 7, 7)  # (bs,768,7,7)

            # Mask image features
            image_embed = self.image_dropout(image_embed)

            # Fuse image and text features
            z = torch.cat([image_embed, text_embed], dim=1)  # (4,768,,)->(4,1536,,)

            # Generate image from the reshaped feature map
            output = self.output_head(z)
            output = output.view(z.size(0), -1, self.output_size, self.output_size)
            loss = self.loss_fn(output, image)
            return {'loss': loss, 'output': output}

        elif self.encoder_name == 'vit_base':
            # Mask image features
            if self.dropout_mode == 'channel':
                image_embed = image_embed.permute(0, 2, 1).contiguous()  # (bs,768,196)
                image_embed = image_embed.view(image_embed.size(0), image_embed.size(1), 14, 14)  # (bs,768,14,14)

                image_embed = self.image_dropout(image_embed)
                image_embed = image_embed.view(image_embed.size(0), image_embed.size(1), -1)
                image_embed = image_embed.permute(0, 2, 1).contiguous()
            else:
                image_embed = self.image_dropout(image_embed)

            # Fuse image and text features
            z = self.fusion_attention(query=image_embed, key=text_embed, value=text_embed)

            # Add positional encoding
            z = z + self.positional_encoding.unsqueeze(0)

            # Transformer decoding
            z = self.transformer_decoder(z, z)  # (bs,196,768)

            # Reshape before feeding into ConvTranspose layers
            z = z.permute(0, 2, 1).contiguous()  # (bs,768,196) (bs,768,49)
            if self.encoder_name == 'resnet_50':
                z = z.view(z.size(0), z.size(1), 7, 7)  # (bs,768,7,7)
            elif self.encoder_name == 'vit_base':
                z = z.view(z.size(0), z.size(1), 14, 14)  # (bs,768,14,14)

            # Generate image from the reshaped feature map
            output = self.output_head(z)
            output = output.view(z.size(0), -1, self.output_size, self.output_size)
            loss = self.loss_fn(output, image)
            return {'loss': loss, 'output': output}


    def loss_fn(self, recon_image, image):
        if image.size(2) != recon_image.size(2) or image.size(3) != recon_image.size(3):
            image = F.interpolate(image, size=recon_image.size()[2:], mode='bilinear', align_corners=True)
        recon_loss = F.l1_loss(recon_image, image)
        return recon_loss * 10