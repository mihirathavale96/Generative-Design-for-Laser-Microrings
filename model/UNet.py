# UNet.py (patched implementation)
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod

# -------------------------
# Helpers
# -------------------------
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# TimestepBlock interface
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

# -------------------------
# Normalization helper
# -------------------------
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

# -------------------------
# Residual / Upsample / Downsample blocks
# -------------------------
class ResidualBlock(TimestepBlock):
	def __init__(self, in_channels, out_channels, time_channels, dropout, residual_scale=1.0):
		super().__init__()
		self.residual_scale = residual_scale
		self.conv1 = nn.Sequential(
			norm_layer(in_channels),
			nn.SiLU(),
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		)
		self.time_emb = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_channels, out_channels)
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
	
	def forward(self, x, t):
		h = self.conv1(x)
		h = h + self.time_emb(t)[:, :, None, None]
		h = self.conv2(h)
		return self.shortcut(x) + self.residual_scale * h

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
# New Attention Blocks
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
		
		# Normalization and projection for image tokens
		self.norm = nn.LayerNorm(channels)
		self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
		self.out_proj = nn.Linear(channels, channels)
	
		self.attn_dropout = nn.Dropout(dropout)
	
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
	Cross-attention block where images attend to parameters and masked objectives
	"""
	def __init__(self, img_channels, param_dim, obj_dim, num_heads=1, dropout=0.1):
		super().__init__()
		self.num_heads = num_heads
		assert img_channels % num_heads == 0
		self.head_dim = img_channels // num_heads
		
		# Image token normalization and query projection
		self.img_norm = nn.LayerNorm(img_channels)
		self.to_q = nn.Linear(img_channels, img_channels, bias=False)
		
		# Parameter and objective projections to match image channels
		self.param_proj = nn.Linear(param_dim, img_channels)
		self.obj_proj = nn.Linear(obj_dim, img_channels)
		
		# Context normalization and key/value projection
		self.context_norm = nn.LayerNorm(img_channels)
		self.to_kv = nn.Linear(img_channels, 2 * img_channels, bias=False)
		
		self.out_proj = nn.Linear(img_channels, img_channels)
	
		self.attn_dropout = nn.Dropout(dropout)
	
	def forward(self, img_tokens, param_tokens, obj_emb=None):
		B, N_img, C = img_tokens.shape
		
		# Query from image tokens
		q = self.to_q(self.img_norm(img_tokens))
		
		# Project parameters to match image channels
		param_tokens_proj = self.param_proj(param_tokens)  # [B, param_dim] -> [B, C]
		param_tokens_proj = param_tokens_proj.unsqueeze(1)  # [B, 1, C]
		
		# Build context (parameters + optionally objectives)
		context = param_tokens_proj  # [B, 1, C]
		
		if obj_emb is not None:
			obj_tokens_proj = self.obj_proj(obj_emb).unsqueeze(1)  # [B, 1, C]
			context = torch.cat([context, obj_tokens_proj], dim=1)  # [B, 2, C]
		
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
# Encoders for parameters & objectives
# -------------------------
class ObjectiveEncoder(nn.Module):
	def __init__(self, obj_dim, hidden_dim, out_dim, time_dim, residual_scale=0.1):
		super().__init__()
		self.obj_dim = obj_dim
		self.residual_scale = residual_scale
		self.time_embed = nn.Sequential(
			nn.Linear(time_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
		)
	
		if obj_dim == 1:
			self.input_proj = nn.Linear(1, hidden_dim)
		else:
			self.input_proj = nn.Sequential(
				nn.Linear(obj_dim, hidden_dim),
				nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim)
			)
	
		self.layers = nn.ModuleList([
			nn.Sequential(
				nn.Linear(hidden_dim, hidden_dim),
				nn.SiLU(),
				nn.Linear(hidden_dim, hidden_dim)
			)
			for _ in range(2)
		])
	
		self.out_proj = nn.Linear(hidden_dim, out_dim)
		self.null_embedding = nn.Parameter(torch.randn(out_dim))
	
	def forward(self, objectives, time_emb, mask=None):
		batch_size = objectives.shape[0]
		if mask is None:
			mask = torch.ones(batch_size, dtype=torch.bool, device=objectives.device)
	
		h = self.input_proj(objectives)
		time_e = self.time_embed(time_emb)
		h = h + time_e
		for layer in self.layers:
			h = layer(h) + self.residual_scale*h
		h = self.out_proj(h)
	
		null_emb = self.null_embedding.unsqueeze(0).expand(batch_size, -1)
		h = torch.where(mask.unsqueeze(1), h, null_emb)
		return h

class ParameterEncoder(nn.Module):
	def __init__(self, param_dim, hidden_dim, out_dim, time_dim, residual_scale=0.1):
		super().__init__()
		self.residual_scale = residual_scale
		self.time_embed = nn.Sequential(
			nn.Linear(time_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
		)
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
	
	def forward(self, params, time_emb):
		h = self.input_proj(params)
		time_e = self.time_embed(time_emb)
		h = h + time_e
		for layer in self.layers:
			h = layer(h) + self.residual_scale*h
		return self.out_proj(h)

# -------------------------
# Sequential container that forwards params and obj_emb
# -------------------------
class TimestepEmbedSequentialWithObjective(nn.Sequential, TimestepBlock):
    """
    Sequential container that routes timestep embedding, params and obj_emb.
    """
    def forward(self, x, emb, params=None, obj_emb=None):
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
                    img_tokens = layer(img_tokens, params, obj_emb)
                else:
                    # fallback: zero param vector matching device and batch
                    zero_params = torch.zeros(x.shape[0], layer.param_dim, device=x.device)
                    img_tokens = layer(img_tokens, zero_params, obj_emb)
                x = img_tokens.permute(0, 2, 1).reshape(B, C, H, W)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x, params

# -------------------------
# UNet definition (patched)
# -------------------------
class UNet(nn.Module):
    def __init__(
            self,
            in_channels=1,
            out_channels=1,
            model_channels=64,
            param_dim=7,
            param_hidden_dim=64,
            obj_dim=2,
            obj_hidden_dim=32,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0.1,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            use_cross_attention=True,
            use_objective_conditioning=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
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
        self.use_objective_conditioning = use_objective_conditioning

        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Parameter encoder
        self.param_encoder = ParameterEncoder(
            param_dim=param_dim,
            hidden_dim=param_hidden_dim,
            out_dim=param_hidden_dim,
            time_dim=time_embed_dim
        )

        # Objective encoder for classifier-free guidance
        if self.use_objective_conditioning:
            self.objective_encoder = ObjectiveEncoder(
                obj_dim=obj_dim,
                hidden_dim=obj_hidden_dim,
                out_dim=obj_hidden_dim,
                time_dim=time_embed_dim
            )

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequentialWithObjective(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                
                # First residual block
                layers.append(ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout))
                ch = mult * model_channels
                
                # Second residual block
                layers.append(ResidualBlock(ch, ch, time_embed_dim, dropout))
                
                if ds in attention_resolutions:
                    # Self-attention for images
                    layers.append(AttentionBlock(ch, num_heads, dropout))

                    # Cross-attention between images and parameters/objectives
                    if self.use_cross_attention:
                        layers.append(CrossAttentionBlock(ch, param_hidden_dim, obj_hidden_dim, num_heads, dropout))

                self.down_blocks.append(TimestepEmbedSequentialWithObjective(*layers))
                down_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequentialWithObjective(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block - following the same Residual->Residual->SelfAttn->CrossAttn pattern
        middle_layers = [
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads, dropout)
        ]

        # Add cross-attention block at bottleneck
        if self.use_cross_attention:
            middle_layers.append(CrossAttentionBlock(ch, param_hidden_dim, obj_hidden_dim, num_heads, dropout))

        self.middle_block = TimestepEmbedSequentialWithObjective(*middle_layers)

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = []
                
                # First residual block
                layers.append(ResidualBlock(
                    ch + down_block_chans.pop(),
                    model_channels * mult,
                    time_embed_dim,
                    dropout
                ))
                ch = model_channels * mult
                
                # Second residual block
                layers.append(ResidualBlock(ch, ch, time_embed_dim, dropout))
                
                if ds in attention_resolutions:
                    # Self-attention for images
                    layers.append(AttentionBlock(ch, num_heads, dropout))

                    # Cross-attention between images and parameters/objectives
                    if self.use_cross_attention:
                        layers.append(CrossAttentionBlock(ch, param_hidden_dim, obj_hidden_dim, num_heads=num_heads))

                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequentialWithObjective(*layers))

        # Output projections
        self.img_out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )
        self.param_out = nn.Linear(param_hidden_dim, param_dim)

    def forward(self, x, params, timesteps, objectives=None, cfg_mask=None):
        """
        :param x: [N, C, H, W] - input images
        :param params: [N, param_dim] - input parameters
        :param timesteps: [N] - diffusion timesteps
        :param objectives: [N, obj_dim] - objective targets (optional)
        :param cfg_mask: [N] - boolean mask for classifier-free guidance (optional)
        :return: (img_out, param_out) - predicted noise for images and parameters
        """
        hs = []

        # Time embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Process parameters
        param_features = self.param_encoder(params, emb)  # [B, param_hidden_dim]

        # Process objectives
        obj_emb = None
        if self.use_objective_conditioning and objectives is not None:
            obj_emb = self.objective_encoder(objectives, emb, cfg_mask)

        # Down stage
        h = x
        for module in self.down_blocks:
            h, param_features = module(h, emb, param_features, obj_emb)
            hs.append(h * 0.1) # scale down skip connections

        # Middle
        h, param_features = self.middle_block(h, emb, param_features, obj_emb)

        # Up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h, param_features = module(cat_in, emb, param_features, obj_emb)

        img_out = self.img_out(h)
        param_out = self.param_out(param_features)
        return img_out, param_out