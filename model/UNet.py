import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod
import warnings


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.
	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
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


# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, params=None):
        for layer in self:
            if isinstance(layer, CrossModalAttentionBlock) and params is not None:
                x = layer(x, params)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # projection for time step embedding
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
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut - IMPROVED with dropout
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        # FIXED: Use standard sqrt scaling instead of double sqrt
        scale = 1. / math.sqrt(C // self.num_heads)
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  # Added dropout for regularization
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


# IMPROVED Cross-Modal Attention Block with richer parameter representation
class CrossModalAttentionBlock(nn.Module):
    def __init__(self, img_channels, param_dim, num_heads=1, n_param_tokens=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.n_param_tokens = n_param_tokens  # Multiple tokens for richer representation
        assert img_channels % num_heads == 0
        
        # Image feature processing
        self.img_norm = norm_layer(img_channels)
        self.img_to_q = nn.Conv2d(img_channels, img_channels, kernel_size=1, bias=False)
        
        # Parameter processing with expanded representation
        self.param_norm = nn.LayerNorm(param_dim)
        # Project parameters to multiple tokens
        self.param_to_kv = nn.Linear(param_dim, img_channels * 2 * n_param_tokens)
        
        # Output projection
        self.proj = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img, params):
        B, C, H, W = img.shape
        
        # Process image to create queries
        img_norm = self.img_norm(img)
        q = self.img_to_q(img_norm)
        q = q.reshape(B * self.num_heads, C // self.num_heads, H * W)  # [B*nh, C/nh, H*W]
        
        # Process parameters to create keys and values with multiple tokens
        param_norm = self.param_norm(params)
        kv = self.param_to_kv(param_norm)  # [B, C*2*n_tokens]
        kv = kv.reshape(B, 2, C, self.n_param_tokens)  # [B, 2, C, n_tokens]
        k, v = kv[:, 0], kv[:, 1]  # Each is [B, C, n_tokens]
        
        # Reshape for multi-head attention
        k = k.reshape(B * self.num_heads, C // self.num_heads, self.n_param_tokens)
        v = v.reshape(B * self.num_heads, C // self.num_heads, self.n_param_tokens)
        
        # Cross-attention with proper scaling
        scale = 1. / math.sqrt(C // self.num_heads)  # Fixed: standard sqrt scaling
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)  # [B*nh, H*W, n_tokens]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  # Added dropout
        
        # Apply attention weights
        h = torch.einsum("bts,bcs->bct", attn, v)  # [B*nh, C/nh, H*W]
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return h + img  # Residual connection


# Objective Conditioning Encoder
class ObjectiveEncoder(nn.Module):
    def __init__(self, obj_dim, hidden_dim, out_dim, time_dim):
        super().__init__()
        self.obj_dim = obj_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Handle both scalar and vector objectives
        if obj_dim == 1:
            # For scalar objectives, use simple embedding
            self.input_proj = nn.Linear(1, hidden_dim)
        else:
            # For vector objectives, use more complex projection
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
        
        # Null conditioning embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.randn(out_dim))
        
    def forward(self, objectives, time_emb, mask=None):
        """
        Args:
            objectives: [B, obj_dim] tensor of objective values
            time_emb: [B, time_dim] tensor of time embeddings
            mask: [B] boolean tensor, True for conditional, False for unconditional
        """
        batch_size = objectives.shape[0]
        
        if mask is None:
            # All conditional
            mask = torch.ones(batch_size, dtype=torch.bool, device=objectives.device)
        
        # Embed objectives
        h = self.input_proj(objectives)
        
        # Add time embedding
        time_emb = self.time_embed(time_emb)
        h = h + time_emb
        
        # Process through layers
        for layer in self.layers:
            h = layer(h) + h  # Residual connection
            
        h = self.out_proj(h)
        
        # Apply classifier-free guidance masking
        # Where mask is False, use null embedding
        null_emb = self.null_embedding.unsqueeze(0).expand(batch_size, -1)
        h = torch.where(mask.unsqueeze(1), h, null_emb)
        
        return h


# IMPROVED Parameter Encoder MLP with better feature extraction
class ParameterEncoder(nn.Module):
    def __init__(self, param_dim, hidden_dim, out_dim, time_dim, num_layers=3):
        super().__init__()
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
            for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, params, time_emb):
        # Embed parameters
        h = self.input_proj(params)
        
        # Add time embedding
        time_emb = self.time_embed(time_emb)
        h = h + time_emb
        
        # Process through layers
        for layer in self.layers:
            h = layer(h) + h  # Residual connection
            
        return self.out_proj(h)


# IMPROVED Cross-Modal Attention Block with Objective Conditioning
class CrossModalObjectiveAttentionBlock(nn.Module):
    def __init__(self, img_channels, param_dim, obj_dim, num_heads=1, n_tokens=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.n_tokens = n_tokens
        assert img_channels % num_heads == 0
        
        # Image feature processing
        self.img_norm = norm_layer(img_channels)
        self.img_to_q = nn.Conv2d(img_channels, img_channels, kernel_size=1, bias=False)
        
        # Combined parameter + objective processing
        combined_dim = param_dim + obj_dim
        self.combined_norm = nn.LayerNorm(combined_dim)
        # Project to multiple tokens for richer representation
        self.combined_to_kv = nn.Linear(combined_dim, img_channels * 2 * n_tokens)
        
        # Output projection
        self.proj = nn.Conv2d(img_channels, img_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img, params, obj_emb):
        B, C, H, W = img.shape
        
        # Process image to create queries
        img_norm = self.img_norm(img)
        q = self.img_to_q(img_norm)
        q = q.reshape(B * self.num_heads, C // self.num_heads, H * W)
        
        # Combine parameters and objective embeddings
        combined = torch.cat([params, obj_emb], dim=-1)
        combined_norm = self.combined_norm(combined)
        kv = self.combined_to_kv(combined_norm)
        kv = kv.reshape(B, 2, C, self.n_tokens)
        k, v = kv[:, 0], kv[:, 1]
        
        # Reshape for attention
        k = k.reshape(B * self.num_heads, C // self.num_heads, self.n_tokens)
        v = v.reshape(B * self.num_heads, C // self.num_heads, self.n_tokens)
        
        # Cross-attention with proper scaling
        scale = 1. / math.sqrt(C // self.num_heads)  # Fixed: standard sqrt scaling
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)  # Added dropout
        
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)
        
        return h + img


# Updated TimestepEmbedSequential to handle objective conditioning
class TimestepEmbedSequentialWithObjective(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings and objective conditioning
    to the children that support it as extra input.
    """

    def forward(self, x, emb, params=None, obj_emb=None):
        for layer in self:
            if isinstance(layer, CrossModalObjectiveAttentionBlock):
                # This layer requires both params and obj_emb
                if params is not None and obj_emb is not None:
                    x = layer(x, params, obj_emb)
                elif params is not None or obj_emb is not None:
                    # Log warning if only one input is provided
                    warnings.warn("CrossModalObjectiveAttentionBlock received incomplete inputs")
            elif isinstance(layer, CrossModalAttentionBlock):
                # This layer requires params
                if params is not None:
                    x = layer(x, params)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
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


# NEW: Parameter Decoder that combines features from both modalities
class ParameterDecoder(nn.Module):
    def __init__(self, param_hidden_dim, img_channels, param_dim, hidden_dim=256):
        super().__init__()
        
        # Combine features from both parameter encoder and image path
        self.combine_features = nn.Sequential(
            nn.Linear(param_hidden_dim + img_channels, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        
    def forward(self, param_features, img_features_pooled):
        combined = torch.cat([param_features, img_features_pooled], dim=1)
        return self.combine_features(combined)


# The IMPROVED UNet model with better cross-modal integration
class UNet(nn.Module):
	def __init__(
			self,
			in_channels=1,
			out_channels=1,
			model_channels=128,
			param_dim=8,
			param_hidden_dim=128,
			obj_dim=1,
			obj_hidden_dim=64,
			num_res_blocks=2,
			attention_resolutions=(8, 16),
			dropout=0.1,
			channel_mult=(1, 2, 2, 2),
			conv_resample=True,
			num_heads=4,
			use_cross_attention=True,
			use_objective_conditioning=True,
			n_param_tokens=4,  # NEW: number of tokens for parameter representation
			normalize_params=True  # NEW: whether to normalize parameters
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
		self.n_param_tokens = n_param_tokens
		self.normalize_params = normalize_params
	
		# NEW: Parameter normalization for stable training
		if normalize_params:
			self.param_scale = nn.Parameter(torch.ones(param_dim))
			self.param_shift = nn.Parameter(torch.zeros(param_dim))
	
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
			time_dim=time_embed_dim,
			num_layers=3
		)
		
		# Objective encoder for classifier-free guidance
		if self.use_objective_conditioning:
			self.objective_encoder = ObjectiveEncoder(
				obj_dim=obj_dim,
				hidden_dim=obj_hidden_dim,
				out_dim=obj_hidden_dim,
				time_dim=time_embed_dim
			)
		
		#### encoder-decoder image+parameter path ####
		
		# down blocks
		self.down_blocks = nn.ModuleList([
			TimestepEmbedSequentialWithObjective(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
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
					# Add self-attention to images with dropout
					layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropout))
					
					# Add cross-attention between image and parameter+objective modalities
					if self.use_cross_attention:
						if self.use_objective_conditioning:
							layers.append(CrossModalObjectiveAttentionBlock(
								ch, param_hidden_dim, obj_hidden_dim, 
								num_heads=num_heads, n_tokens=n_param_tokens, dropout=dropout
							))
						else:
							layers.append(CrossModalAttentionBlock(
								ch, param_hidden_dim, 
								num_heads=num_heads, n_param_tokens=n_param_tokens, dropout=dropout
							))
						
				self.down_blocks.append(TimestepEmbedSequentialWithObjective(*layers))
				down_block_chans.append(ch)
			if level != len(channel_mult) - 1:
				self.down_blocks.append(TimestepEmbedSequentialWithObjective(Downsample(ch, conv_resample)))
				down_block_chans.append(ch)
				ds *= 2
	
		# middle block
		middle_layers = [
			ResidualBlock(ch, ch, time_embed_dim, dropout),
			AttentionBlock(ch, num_heads=num_heads, dropout=dropout)
		]
		
		# Add a cross-attention block at the bottleneck
		if self.use_cross_attention:
			if self.use_objective_conditioning:
				middle_layers.append(CrossModalObjectiveAttentionBlock(
					ch, param_hidden_dim, obj_hidden_dim, 
					num_heads=num_heads, n_tokens=n_param_tokens, dropout=dropout
				))
			else:
				middle_layers.append(CrossModalAttentionBlock(
					ch, param_hidden_dim, 
					num_heads=num_heads, n_param_tokens=n_param_tokens, dropout=dropout
				))
			
		middle_layers.append(ResidualBlock(ch, ch, time_embed_dim, dropout))
		
		self.middle_block = TimestepEmbedSequentialWithObjective(*middle_layers)
	
		# up blocks
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
					# Add self-attention to images
					layers.append(AttentionBlock(ch, num_heads=num_heads, dropout=dropout))
					
					# Add cross-attention between image and parameter+objective modalities
					if self.use_cross_attention:
						if self.use_objective_conditioning:
							layers.append(CrossModalObjectiveAttentionBlock(
								ch, param_hidden_dim, obj_hidden_dim,
								num_heads=num_heads, n_tokens=n_param_tokens, dropout=dropout
							))
						else:
							layers.append(CrossModalAttentionBlock(
								ch, param_hidden_dim,
								num_heads=num_heads, n_param_tokens=n_param_tokens, dropout=dropout
							))
						
				if level and i == num_res_blocks:
					layers.append(Upsample(ch, conv_resample))
					ds //= 2
				self.up_blocks.append(TimestepEmbedSequentialWithObjective(*layers))
	
		#### encoder-decoder image+parameter path ####
	
		# Output projection for image
		self.img_out = nn.Sequential(
			norm_layer(ch),
			nn.SiLU(),
			nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
		)
	
		# IMPROVED: Parameter decoder that combines features from both paths
		self.param_decoder = ParameterDecoder(
			param_hidden_dim=param_hidden_dim,
			img_channels=model_channels,  # ch at the end of up_blocks
			param_dim=param_dim,
			hidden_dim=256
		)
	
	def forward(self, x, params, timesteps, objectives=None, cfg_mask=None):
		"""
		Apply the model to an input batch with optional objective conditioning.
		:param x: an [N x C x H x W] Tensor of image inputs.
		:param params: an [N x param_dim] Tensor of parameter inputs.
		:param timesteps: a 1-D batch of timesteps.
		:param objectives: an [N x obj_dim] Tensor of objective values (optional).
		:param cfg_mask: an [N] boolean Tensor for classifier-free guidance (optional).
		:return: tuple of:
				- image noise prediction [N x C x H x W]
				- parameter noise prediction [N x param_dim]
		"""
		hs = []
		
		# NEW: Normalize parameters if enabled
		if self.normalize_params:
			params = (params - self.param_shift) / (self.param_scale + 1e-5)
		
		# Time step embedding
		emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
		
		# Process parameters
		param_features = self.param_encoder(params, emb)
		
		# Process objectives if using conditioning
		obj_emb = None
		if self.use_objective_conditioning and objectives is not None:
			obj_emb = self.objective_encoder(objectives, emb, cfg_mask)
		
		# Down stage for image
		h = x
		for module in self.down_blocks:
			h = module(h, emb, param_features, obj_emb)
			hs.append(h)
			
		# Middle stage
		h = self.middle_block(h, emb, param_features, obj_emb)
		
		# Up stage
		for module in self.up_blocks:
			cat_in = torch.cat([h, hs.pop()], dim=1)
			h = module(cat_in, emb, param_features, obj_emb)
			
		# Output projections
		img_out = self.img_out(h)
		
		# IMPROVED: Combine features from both paths for parameter prediction
		# Global average pooling on image features
		img_features_pooled = h.mean(dim=[2, 3])  # [B, C]
		param_out = self.param_decoder(param_features, img_features_pooled)
		
		# NEW: Denormalize parameter predictions if normalization was used
		if self.normalize_params:
			param_out = param_out * self.param_scale + self.param_shift
		
		return img_out, param_out