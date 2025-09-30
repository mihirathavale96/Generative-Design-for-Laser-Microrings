# UNetPredictor.py
import math
import torch
from torch import nn
from torch.nn import functional as F

# -------------------------
# Normalization helper
# -------------------------
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

# -------------------------
# Residual / Downsample blocks (without timestep embedding)
# -------------------------
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, dropout, residual_scale=1.0):
		super().__init__()
		self.residual_scale = residual_scale
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
		h = self.conv1(x)
		h = self.conv2(h)
		return self.shortcut(x) + self.residual_scale * h

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

# -------------------------
# Attention Blocks (without timestep embedding)
# -------------------------
class AttentionBlock(nn.Module):
	"""
	Self-attention block where image attends to itself
	"""
	def __init__(self, channels, num_heads=1, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		assert channels % num_heads == 0
		self.head_dim = channels // num_heads
		self.attn_dropout = nn.Dropout(dropout)
		
		# Normalization and projection for image tokens
		self.norm = nn.LayerNorm(channels)
		self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
		self.out_proj = nn.Linear(channels, channels)
	
	def forward(self, img_tokens):
		B, N, C = img_tokens.shape
		
		# Self-attention on image tokens
		qkv = self.to_qkv(self.norm(img_tokens))
		q, k, v = qkv.chunk(3, dim=2)
		
		# Reshape for multi-head attention
		q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
		
		# Attention
		scale = 1.0 / math.sqrt(self.head_dim)
		attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
		attn = self.attn_dropout(attn)
		out = torch.matmul(attn, v)
		
		# Reshape back
		out = out.transpose(1, 2).contiguous().view(B, N, C)
		out = self.out_proj(out)
		
		return out + img_tokens

class CrossAttentionBlock(nn.Module):
	"""
	Cross-attention block where images attend to parameters
	"""
	def __init__(self, img_channels, param_dim, num_heads=1, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		self.param_dim = param_dim
		assert img_channels % num_heads == 0
		self.head_dim = img_channels // num_heads
		self.attn_dropout = nn.Dropout(dropout)
		
		# Image token normalization and query projection
		self.img_norm = nn.LayerNorm(img_channels)
		self.to_q = nn.Linear(img_channels, img_channels, bias=False)
		
		# Parameter projection to match image channels
		self.param_proj = nn.Linear(param_dim, img_channels)
		
		# Context normalization and key/value projection
		self.context_norm = nn.LayerNorm(img_channels)
		self.to_kv = nn.Linear(img_channels, 2 * img_channels, bias=False)
		
		self.out_proj = nn.Linear(img_channels, img_channels)
	
	def forward(self, img_tokens, param_tokens):
		B, N_img, C = img_tokens.shape
		
		# Query from image tokens
		q = self.to_q(self.img_norm(img_tokens))
		
		# Project parameters to match image channels
		param_tokens_proj = self.param_proj(param_tokens)  # [B, param_dim] -> [B, C]
		param_tokens_proj = param_tokens_proj.unsqueeze(1)  # [B, 1, C]
		
		# Context is just parameters (no objectives in predictor)
		context = param_tokens_proj  # [B, 1, C]
		
		# Key/Value from context
		kv = self.to_kv(self.context_norm(context))
		k, v = kv.chunk(2, dim=2)
		
		# Expand context to match image sequence length for cross-attention
		N_context = context.shape[1]
		k = k.unsqueeze(2).expand(-1, -1, N_img, -1).reshape(B, N_context * N_img, C)
		v = v.unsqueeze(2).expand(-1, -1, N_img, -1).reshape(B, N_context * N_img, C)
		
		# Reshape queries for multi-head attention
		q = q.view(B, N_img, self.num_heads, self.head_dim).transpose(1, 2)
		k = k.view(B, N_context * N_img, self.num_heads, self.head_dim).transpose(1, 2)
		v = v.view(B, N_context * N_img, self.num_heads, self.head_dim).transpose(1, 2)
		
		# Cross-attention
		scale = 1.0 / math.sqrt(self.head_dim)
		attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
		attn = self.attn_dropout(attn)
		out = torch.matmul(attn, v)
		
		# Reshape back
		out = out.transpose(1, 2).contiguous().view(B, N_img, C)
		out = self.out_proj(out)
		
		return out + img_tokens

# -------------------------
# Parameter encoder (without time embedding)
# -------------------------
class ParameterEncoder(nn.Module):
	def __init__(self, param_dim, hidden_dim, out_dim, residual_scale=0.1):
		super().__init__()
		self.residual_scale = residual_scale
		self.input_proj = nn.Linear(param_dim, hidden_dim)
		self.layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim),
				nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim)
			)
			for _ in range(2)
		])
		self.out_proj = nn.Linear(hidden_dim, out_dim)
	
	def forward(self, params):
		h = self.input_proj(params)
		for layer in self.layers:
			h = layer(h) + self.residual_scale*h
		return self.out_proj(h)

# -------------------------
# Sequential container for predictor
# -------------------------
class PredictorSequential(nn.Sequential):
    """
    Sequential container that routes params through attention layers
    """
    def forward(self, x, params=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                # Convert image to tokens and apply self-attention
                B, C, H, W = x.shape
                img_tokens = x.reshape(B, C, H*W).permute(0, 2, 1)
                img_tokens = layer(img_tokens)
                x = img_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            elif isinstance(layer, CrossAttentionBlock):
                # Convert image to tokens and apply cross-attention
                B, C, H, W = x.shape
                img_tokens = x.reshape(B, C, H*W).permute(0, 2, 1)
                if params is not None:
                    img_tokens = layer(img_tokens, params)
                else:
                    # fallback: zero param vector matching device and batch
                    zero_params = torch.zeros(x.shape[0], layer.param_dim, device=x.device)
                    img_tokens = layer(img_tokens, zero_params)
                x = img_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            else:
                x = layer(x)
        return x

# -------------------------
# UNetPredictor definition
# -------------------------
class UNetPredictor(nn.Module):
    def __init__(
            self,
            in_channels=1,
            model_channels=64,
            param_dim=7,
            param_hidden_dim=64,
            obj_dim=2,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0.1,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            use_cross_attention=True
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

        # Parameter encoder (no time embedding)
        self.param_encoder = ParameterEncoder(
            param_dim=param_dim,
            hidden_dim=param_hidden_dim,
            out_dim=param_hidden_dim
        )

        # down blocks
        self.down_blocks = nn.ModuleList([
            PredictorSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
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
                    # Self-attention for images
                    layers.append(AttentionBlock(ch, num_heads, dropout))

                    # Cross-attention between images and parameters
                    if self.use_cross_attention:
                        layers.append(CrossAttentionBlock(ch, param_hidden_dim, num_heads, dropout))

                self.down_blocks.append(PredictorSequential(*layers))
                
            if level != len(channel_mult) - 1:
                self.down_blocks.append(PredictorSequential(Downsample(ch, conv_resample)))
                ds *= 2

        # middle block
        middle_layers = [
            ResidualBlock(ch, ch, dropout),
            ResidualBlock(ch, ch, dropout),
            AttentionBlock(ch, num_heads, dropout)
        ]

        # Add cross-attention block at bottleneck
        if self.use_cross_attention:
            middle_layers.append(CrossAttentionBlock(ch, param_hidden_dim, num_heads, dropout))

        self.middle_block = PredictorSequential(*middle_layers)

        # Global average pooling and objective prediction head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Combine image features with parameter features for final prediction
        self.obj_predictor = nn.Sequential(
            nn.Linear(ch + param_hidden_dim, ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ch, ch // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ch // 2, obj_dim)
        )

    def forward(self, x, params):
        """
        :param x: [N, C, H, W] - input images
        :param params: [N, param_dim] - input parameters
        :return: objectives [N, obj_dim] - predicted objectives
        """
        
        # Process parameters
        param_features = self.param_encoder(params)  # [B, param_hidden_dim]

        # Down stage
        h = x
        for module in self.down_blocks:
            h = module(h, param_features)

        # Middle
        h = self.middle_block(h, param_features)

        # Global pooling to get image features
        img_features = self.global_pool(h).squeeze(-1).squeeze(-1)  # [B, ch]
        
        # Combine image and parameter features
        combined_features = torch.cat([img_features, param_features], dim=1)  # [B, ch + param_hidden_dim]
        
        # Predict objectives
        objectives = self.obj_predictor(combined_features)  # [B, obj_dim]
        
        return objectives