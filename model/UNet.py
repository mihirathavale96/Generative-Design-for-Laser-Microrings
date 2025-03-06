import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
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

class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""
        raise NotImplementedError

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to the children that support it."""
    def forward(self, x, emb, params=None):
        for layer in self:
            if isinstance(layer, CrossAttentionBlock) and params is not None:
                x = layer(x, params)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def norm_layer(channels):
    return nn.GroupNorm(32, channels)

class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
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
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

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

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, param_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = norm_layer(channels)
        self.norm_params = nn.LayerNorm(param_dim)
        
        head_dim = channels // num_heads
        
        # Image query projection
        self.q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Parameters key and value projections
        self.k = nn.Linear(param_dim, channels)
        self.v = nn.Linear(param_dim, channels)
        
        # Output projection
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, params):
        B, C, H, W = x.shape
        params = self.norm_params(params)

        # Generate queries from image features
        q = self.q(self.norm(x))
        q = q.reshape(B * self.num_heads, C // self.num_heads, H * W)
        
        # Generate keys and values from parameters
        k = self.k(params)
        v = self.v(params)
        
        # Reshape k and v for attention
        k = k.reshape(B * self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B * self.num_heads, C // self.num_heads, -1)
        
        # Compute scaled dot-product attention
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return h + x

class ParameterEmbedding(nn.Module):
    def __init__(self, param_dim, embed_dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, params):
        return self.embedding(params)

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

class UNet(nn.Module):
	"""
	A UNet model that reconstructs image and parameter inputs.
	Includes self-attention and cross-attention.
	"""
	def __init__(
		self,
		in_channels=1,
		model_channels=64,
		out_channels=1,
		param_dim=8,
		num_res_blocks=2,
		attention_resolutions=(8, 16),
		dropout=0.1,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		num_heads=4
	):
		super().__init__()
	
		self.in_channels = in_channels
		self.model_channels = model_channels
		self.out_channels = out_channels
		self.param_dim = param_dim
		self.num_res_blocks = num_res_blocks
		self.attention_resolutions = attention_resolutions
		self.dropout = dropout
		self.channel_mult = channel_mult
		self.conv_resample = conv_resample
		self.num_heads = num_heads
	
		# Parameter embedding
		param_embed_dim = model_channels * 4
		self.param_embed = ParameterEmbedding(param_dim, param_embed_dim)
	
		# Time embedding
		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			nn.Linear(model_channels, time_embed_dim),
			nn.SiLU(),
			nn.Linear(time_embed_dim, time_embed_dim),
		)
	
		# Down blocks
		self.down_blocks = nn.ModuleList([
			TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
		])
		down_block_chans = [model_channels]
		ch = model_channels
		ds = 1
		
		for level, mult in enumerate(channel_mult):
			for _ in range(num_res_blocks):
				layers = [
					ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
				]
				ch = mult * model_channels
				if ds in attention_resolutions:
					# Add self-attention
					layers.append(AttentionBlock(ch, num_heads=num_heads))
					# Add cross-attention with parameters
					layers.append(CrossAttentionBlock(ch, param_embed_dim, num_heads=num_heads))
				self.down_blocks.append(TimestepEmbedSequential(*layers))
				down_block_chans.append(ch)
			if level != len(channel_mult) - 1:
				self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
				down_block_chans.append(ch)
				ds *= 2
	
		# Middle block
		self.middle_block = TimestepEmbedSequential(
			ResidualBlock(ch, ch, time_embed_dim, dropout),
			AttentionBlock(ch, num_heads=num_heads),
			CrossAttentionBlock(ch, param_embed_dim, num_heads=num_heads),
			ResidualBlock(ch, ch, time_embed_dim, dropout)
		)
	
		# Up blocks
		self.up_blocks = nn.ModuleList([])
		for level, mult in list(enumerate(channel_mult))[::-1]:
			for i in range(num_res_blocks + 1):
				layers = [
					ResidualBlock(
						ch + down_block_chans.pop(),
						model_channels * mult,
						time_embed_dim,
						dropout
					)
				]
				ch = model_channels * mult
				if ds in attention_resolutions:
					# Add self-attention
					layers.append(AttentionBlock(ch, num_heads=num_heads))
					# Add cross-attention with parameters
					layers.append(CrossAttentionBlock(ch, param_embed_dim, num_heads=num_heads))
				if level and i == num_res_blocks:
					layers.append(Upsample(ch, conv_resample))
					ds //= 2
				self.up_blocks.append(TimestepEmbedSequential(*layers))
	
		# Output layers for images, parameters and property
		self.out_image = nn.Sequential(
			norm_layer(ch),
			nn.SiLU(),
			nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
		)
		
		self.out_params = nn.Sequential(
			norm_layer(ch),
			nn.SiLU(),
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(ch, param_dim)
		)
	
	def forward(self, x, timesteps, params):
		"""
		Apply the model to an input batch.
		:param x: an [N x 1 x 128 x 128] Tensor of image inputs.
		:param timesteps: a 1-D batch of timesteps.
		:param params: an [N x param_dim] Tensor of parameter inputs.
		:return: a tuple of (image_output, param_output), with shapes [N x 1 x 128 x 128] and [N x param_dim].
		"""
		hs = []
		
		# Embed time steps
		t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		
		# Embed parameters
		p_emb = self.param_embed(params)
		
		# Down stage
		h = x
		for module in self.down_blocks:
			h = module(h, t_emb, p_emb)
			hs.append(h)
		
		# Middle stage
		h = self.middle_block(h, t_emb, p_emb)

		# Up stage
		for module in self.up_blocks:
			cat_in = torch.cat([h, hs.pop()], dim=1)
			h = module(cat_in, t_emb, p_emb)
		
		image_output = self.out_image(h)
		param_output = self.out_params(h)
		
		return image_output, param_output

class UNet_Energy(nn.Module):
	"""
	Modified UNet from above, that outputs predicted property.
	It is basically half the architecture (only downsampling and middle block).
	"""
	def __init__(
		self,
		in_channels=1,
		model_channels=64,
		param_dim=8,
		prop_dim=1,
		num_res_blocks=2,
		attention_resolutions=(8, 16),
		dropout=0.1,
		channel_mult=(1, 2, 4, 8),
		conv_resample=True,
		num_heads=4
	):
		super().__init__()
		
		self.in_channels = in_channels
		self.model_channels = model_channels
		self.param_dim = param_dim
		self.prop_dim = prop_dim
		
		# Parameter embedding
		param_embed_dim = model_channels * 4
		self.param_embed = ParameterEmbedding(param_dim, param_embed_dim)
		
		# Time embedding
		time_embed_dim = model_channels * 4
		self.time_embed = nn.Sequential(
			nn.Linear(model_channels, time_embed_dim),
			nn.SiLU(),
			nn.Linear(time_embed_dim, time_embed_dim),
		)
		
		# Down blocks
		self.down_blocks = nn.ModuleList([
			TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
		])
		ch = model_channels
		ds = 1
		
		for level, mult in enumerate(channel_mult):
			for _ in range(num_res_blocks):
				layers = [
					ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
				]
				ch = mult * model_channels
				if ds in attention_resolutions:
					layers.append(AttentionBlock(ch, num_heads=num_heads))
					layers.append(CrossAttentionBlock(ch, param_embed_dim, num_heads=num_heads))
				self.down_blocks.append(TimestepEmbedSequential(*layers))
			if level != len(channel_mult) - 1:
				self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
				ds *= 2
		
		# Middle block
		self.middle_block = TimestepEmbedSequential(
			ResidualBlock(ch, ch, time_embed_dim, dropout),
			AttentionBlock(ch, num_heads=num_heads),
			CrossAttentionBlock(ch, param_embed_dim, num_heads=num_heads),
			ResidualBlock(ch, ch, time_embed_dim, dropout)
		)
		
		# Property predictor MLP
		self.property_predictor = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),  # Global average pooling
			nn.Flatten(),
			nn.Linear(ch, 64),
			nn.SiLU(),
			nn.Linear(64, prop_dim)
		)
	
	def forward(self, x, timesteps, params):
		# Time embedding
		t_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		# Parameter embedding
		p_emb = self.param_embed(params)
		
		# Down blocks
		h = x
		for module in self.down_blocks:
			h = module(h, t_emb, p_emb)
		
		# Middle block
		h = self.middle_block(h, t_emb, p_emb)
		
		# Property prediction
		return self.property_predictor(h)