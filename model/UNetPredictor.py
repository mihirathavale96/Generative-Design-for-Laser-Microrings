import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Standard Residual block (no time embedding)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        """
        h = self.conv1(x)
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# Cross-Modal Attention Block for joint image-parameter processing
class CrossModalAttentionBlock(nn.Module):
    def __init__(self, img_channels, param_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert img_channels % num_heads == 0
        
        # Image feature processing
        self.img_norm = norm_layer(img_channels)
        self.img_to_q = nn.Conv2d(img_channels, img_channels, kernel_size=1, bias=False)
        
        # Parameter processing
        self.param_norm = nn.LayerNorm(param_dim)
        self.param_to_kv = nn.Linear(param_dim, img_channels * 2)
        
        # Output projection
        self.proj = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        
    def forward(self, img, params):
        B, C, H, W = img.shape
        
        # Process image to create queries
        img_norm = self.img_norm(img)
        q = self.img_to_q(img_norm)
        q = q.reshape(B * self.num_heads, C // self.num_heads, H * W)  # [B*nh, C/nh, H*W]
        
        # Process parameters to create keys and values
        param_norm = self.param_norm(params)
        kv = self.param_to_kv(param_norm)  # [B, img_channels*2]
        k, v = kv.chunk(2, dim=1)
        
        # Reshape k, v for attention
        k = k.unsqueeze(-1)  # [B, C, 1]
        v = v.unsqueeze(-1)  # [B, C, 1]
        
        # Reshape for multi-head attention
        k = k.reshape(B * self.num_heads, C // self.num_heads, 1)  # [B*nh, C/nh, 1]
        v = v.reshape(B * self.num_heads, C // self.num_heads, 1)  # [B*nh, C/nh, 1]
        
        # Cross-attention
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # [B*nh, H*W, 1]
        attn = attn.softmax(dim=1)
        
        # Apply attention weights
        h = torch.einsum("bts,bcs->bct", attn, v)  # [B*nh, C/nh, H*W]
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return h + img  # Residual connection


# Parameter Encoder MLP (no time embedding)
class ParameterEncoder(nn.Module):
    def __init__(self, param_dim, hidden_dim, out_dim):
        super().__init__()
        self.input_proj = nn.Linear(param_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(3)  # 3 layers of transformation
        ])
        
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, params):
        # Embed parameters
        h = self.input_proj(params)
        
        # Process through layers
        for layer in self.layers:
            h = layer(h) + h  # Residual connection
            
        return self.out_proj(h)


# Sequential module for parameter passing
class SequentialWithParams(nn.Sequential):
    """
    A sequential module that passes parameters to cross-modal attention blocks.
    """

    def forward(self, x, params=None):
        for layer in self:
            if isinstance(layer, CrossModalAttentionBlock) and params is not None:
                x = layer(x, params)
            else:
                x = layer(x)
        return x


# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        return self.op(x)


# Property Prediction Encoder (Encoder-only architecture)
class PropertyPredictionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        model_channels=128,
        param_dim=8,
        param_hidden_dim=128,
        obj_dim=1,  # dimension of properties to predict
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        conv_resample=True,
        num_heads=4,
        use_cross_attention=True,
        global_pool='adaptive'  # 'adaptive', 'avg', or 'max'
    ):
        super().__init__()
    
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.param_dim = param_dim
        self.obj_dim = obj_dim
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention
        self.global_pool = global_pool
    
        # Parameter encoder
        self.param_encoder = ParameterEncoder(
            param_dim=param_dim,
            hidden_dim=param_hidden_dim,
            out_dim=param_hidden_dim
        )
        
        #### encoder path only ####
        
        # down blocks
        self.encoder_blocks = nn.ModuleList([
            SequentialWithParams(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    # Add self-attention to images
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                    
                    # Add cross-attention between image and parameter modalities
                    if self.use_cross_attention:
                        layers.append(CrossModalAttentionBlock(ch, param_hidden_dim, num_heads=num_heads))
                        
                self.encoder_blocks.append(SequentialWithParams(*layers))
                
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(SequentialWithParams(Downsample(ch, conv_resample)))
                ds *= 2
    
        # bottleneck block
        bottleneck_layers = [
            ResidualBlock(ch, ch, dropout),
            AttentionBlock(ch, num_heads=num_heads)
        ]
        
        # Add a cross-attention block at the bottleneck
        if self.use_cross_attention:
            bottleneck_layers.append(CrossModalAttentionBlock(ch, param_hidden_dim, num_heads=num_heads))
            
        bottleneck_layers.append(ResidualBlock(ch, ch, dropout))
        
        self.bottleneck_block = SequentialWithParams(*bottleneck_layers)
        
        # Store final channel dimension for fusion
        self.final_channels = ch
    
        #### Feature fusion and prediction head ####
        
        # Global pooling layer for image features
        if global_pool == 'adaptive':
            self.global_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif global_pool == 'avg':
            self.global_pool_layer = nn.AdaptiveAvgPool2d(1)
        elif global_pool == 'max':
            self.global_pool_layer = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported global_pool: {global_pool}")
        
        # Feature fusion layer
        combined_dim = self.final_channels + param_hidden_dim
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # Property prediction head
        self.property_head = nn.Linear(combined_dim // 2, obj_dim)
    
    def forward(self, x, params):
        """
        Apply the model to predict properties from images and parameters.
        :param x: an [N x C x H x W] Tensor of image inputs.
        :param params: an [N x param_dim] Tensor of parameter inputs.
        :return: an [N x obj_dim] Tensor of predicted properties.
        """
        # Process parameters
        param_features = self.param_encoder(params)
        
        # Encoder stage for image
        h = x
        for module in self.encoder_blocks:
            h = module(h, param_features)
            
        # Bottleneck stage
        h = self.bottleneck_block(h, param_features)
        
        # Global pooling of image features
        # h: [N, C, H, W] -> [N, C, 1, 1] -> [N, C]
        img_global = self.global_pool_layer(h).squeeze(-1).squeeze(-1)
        
        # Combine image and parameter features
        combined_features = torch.cat([img_global, param_features], dim=1)
        
        # Feature fusion
        fused_features = self.fusion_layers(combined_features)
        
        # Predict properties
        properties = self.property_head(fused_features)
        
        return properties