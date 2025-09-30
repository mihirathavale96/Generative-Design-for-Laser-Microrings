from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class GaussianDiffusionTrainer(nn.Module):
	def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int, 
				 img_weight: float = 2.0, param_weight: float = 1.0, loss_scale: float = 100.0,
				 cfg_dropout_prob: float = 0.2):
		super().__init__()
		self.model = model
		self.T = T
		self.img_weight = img_weight
		self.param_weight = param_weight
		self.cfg_dropout_prob = cfg_dropout_prob
		self.loss_scale = loss_scale
	
		# generate T steps of beta
		self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
	
		# calculate the cumulative product of α, named ᾱ_t in paper
		alpha_t = 1.0 - self.beta_t
		alpha_t_bar = torch.cumprod(alpha_t, dim=0)
	
		# calculate and store two coefficient of q(x_t | x_0)
		self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
		self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))
	
	def forward(self, x_0, params_0, objectives):
		"""
		Train the simplified diffusion model to denoise both image and parameters.
		
		Args:
			x_0: Clean images [batch, 1, 128, 128]
			params_0: Clean parameters [batch, n_params]
			objectives: Target objective values [batch, obj_dim]
		"""
		batch_size = x_0.shape[0]
		
		# Get a random training step t ~ Uniform({1, ..., T})
		t = torch.randint(0, self.T, size=(batch_size,), device=x_0.device)
		
		# Generate noise for both image and parameters
		img_noise = torch.randn_like(x_0)
		params_noise = torch.randn_like(params_0)
		
		# Add noise to the image at timestep t
		x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
			   extract(self.noise_rate, t, x_0.shape) * img_noise)
		
		# Add noise to the parameters at timestep t
		params_signal_rate = extract(self.signal_rate, t, params_0.shape)
		params_noise_rate = extract(self.noise_rate, t, params_0.shape)
		params_t = params_signal_rate * params_0 + params_noise_rate * params_noise
		
		# Create classifier-free guidance mask
		# Randomly drop conditioning for classifier-free guidance
		cfg_mask = torch.rand(batch_size, device=x_0.device) > self.cfg_dropout_prob
		
		# Predict noise of image and parameters
		img_pred, params_pred = self.model(x_t, params_t, t, objectives, cfg_mask)
		
		# Calculate loss for image and parameters
		img_loss = F.mse_loss(img_pred, img_noise, reduction="mean")
		params_loss = F.mse_loss(params_pred, params_noise, reduction="mean")
		
		# Weighted total loss
		total_loss = self.img_weight * img_loss + self.param_weight * params_loss
		
		return total_loss * self.loss_scale #, img_loss, params_loss
	

class PropertyPredictionTrainer(nn.Module):
    """Unchanged - keeping for backward compatibility"""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, x_0, params_0, prop_0):
        """
        Train the predictor
        
        Args:
            x_0: Clean images [batch, 1, 128, 128]
            params_0: Clean parameters [batch, n_params]
            prop_0: Objective value [batch, n_obj]
        """
        prop_pred = self.model(x_0, params_0)
        prop_loss = F.mse_loss(prop_pred, prop_0, reduction="mean")
        return prop_loss


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T
    
        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
    
        # calculate the cumulative product of α, named ᾱ_t in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)
    
        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))
    
    @torch.no_grad()
    def predict_noise(self, x_t, params_t, t, objectives, cfg_scale=1.0):
        """
        Predict noise with optional classifier-free guidance
        """
        if cfg_scale != 1.0:
            # Classifier-free guidance: predict both conditional and unconditional
            batch_size = x_t.shape[0]
            
            # Duplicate inputs for conditional and unconditional predictions
            x_t_doubled = torch.cat([x_t, x_t], dim=0)
            params_t_doubled = torch.cat([params_t, params_t], dim=0)
            t_doubled = torch.cat([t, t], dim=0)
            objectives_doubled = torch.cat([objectives, objectives], dim=0)
            
            # Create CFG mask: first half conditional (True), second half unconditional (False)
            cfg_mask = torch.cat([
                torch.ones(batch_size, dtype=torch.bool, device=x_t.device),
                torch.zeros(batch_size, dtype=torch.bool, device=x_t.device)
            ])
            
            # Get predictions
            img_noise_pred_doubled, params_noise_pred_doubled = self.model(
                x_t_doubled, params_t_doubled, t_doubled, objectives_doubled, cfg_mask
            )
            
            # Split conditional and unconditional predictions
            img_noise_cond, img_noise_uncond = img_noise_pred_doubled.chunk(2, dim=0)
            params_noise_cond, params_noise_uncond = params_noise_pred_doubled.chunk(2, dim=0)
            
            # Apply classifier-free guidance
            img_noise_pred = img_noise_uncond + cfg_scale * (img_noise_cond - img_noise_uncond)
            params_noise_pred = params_noise_uncond + cfg_scale * (params_noise_cond - params_noise_uncond)
        else:
            # Standard prediction without CFG
            cfg_mask = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
            img_noise_pred, params_noise_pred = self.model(x_t, params_t, t, objectives, cfg_mask)
        
        return img_noise_pred, params_noise_pred
    
    @torch.no_grad()
    def sample_one_step(self, x_t, params_t, time_step: int, objectives, cfg_scale=1.0):
        """
        Calculate x_{t-1} and params_{t-1} according to x_t and params_t
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        
        # Predict noise
        img_noise_pred, params_noise_pred = self.predict_noise(x_t, params_t, t, objectives, cfg_scale)
        
        # Calculate mean for both image and parameters
        img_mean = (extract(self.coeff_1, t, x_t.shape) * x_t - 
                   extract(self.coeff_2, t, x_t.shape) * img_noise_pred)
        params_mean = (extract(self.coeff_1, t, params_t.shape) * params_t - 
                      extract(self.coeff_2, t, params_t.shape) * params_noise_pred)
        
        # Get variance
        img_var = extract(self.posterior_variance, t, x_t.shape)
        params_var = extract(self.posterior_variance, t, params_t.shape)
        
        # Sample noise for both image and parameters if time_step > 0
        img_z = torch.randn_like(x_t) if time_step > 0 else 0
        params_z = torch.randn_like(params_t) if time_step > 0 else 0
        
        # Calculate x_{t-1} and params_{t-1}
        x_t_minus_one = img_mean + torch.sqrt(img_var) * img_z
        params_t_minus_one = params_mean + torch.sqrt(params_var) * params_z
    
        # Check for NaN values
        if torch.isnan(x_t_minus_one).any() or torch.isnan(params_t_minus_one).any():
            raise ValueError("NaN detected in sampling!")
    
        return x_t_minus_one, params_t_minus_one
    
    @torch.no_grad()
    def forward(self, x_t, params_t, objectives, cfg_scale=1.0, 
                only_return_x_0: bool = True, interval: int = 1, **kwargs):
        """
        Sample from the simplified diffusion model.
        
        Parameters:
            x_t: Standard Gaussian noise for images [batch_size, channels, height, width]
            params_t: Standard Gaussian noise for parameters [batch_size, n_params]
            objectives: Target objective values [batch_size, obj_dim]
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            only_return_x_0: If True, only return final results
            interval: Interval for saving intermediate steps
            
        Returns:
            If only_return_x_0=True: Tuple of (image, parameters)
            If only_return_x_0=False: Tuple of (image_sequence, parameters_sequence)
        """
        img_samples = [x_t] if not only_return_x_0 else []
        param_samples = [params_t] if not only_return_x_0 else []
        
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T, desc="DDPM Sampling") as pbar:
            for time_step in pbar:
                x_t, params_t = self.sample_one_step(x_t, params_t, time_step, objectives, cfg_scale)
    
                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    img_samples.append(torch.clip(x_t, -1.0, 1.0))
                    param_samples.append(params_t)
    
                pbar.set_postfix({
                    "step": time_step + 1, 
                    "cfg_scale": f"{cfg_scale:.1f}"
                })
    
        if only_return_x_0:
            return torch.clip(x_t, -1.0, 1.0), params_t
        
        # Stack all samples along a new dimension
        img_sequence = torch.stack(img_samples, dim=1)
        param_sequence = torch.stack(param_samples, dim=1)
        
        return img_sequence, param_sequence


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of α, named ᾱ_t in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def predict_noise(self, x_t, params_t, t, objectives, cfg_scale=1.0):
        """
        Predict noise with optional classifier-free guidance
        """
        if cfg_scale != 1.0:
            # Classifier-free guidance: predict both conditional and unconditional
            batch_size = x_t.shape[0]
            
            # Duplicate inputs for conditional and unconditional predictions
            x_t_doubled = torch.cat([x_t, x_t], dim=0)
            params_t_doubled = torch.cat([params_t, params_t], dim=0)
            t_doubled = torch.cat([t, t], dim=0)
            objectives_doubled = torch.cat([objectives, objectives], dim=0)
            
            # Create CFG mask: first half conditional (True), second half unconditional (False)
            cfg_mask = torch.cat([
                torch.ones(batch_size, dtype=torch.bool, device=x_t.device),
                torch.zeros(batch_size, dtype=torch.bool, device=x_t.device)
            ])
            
            # Get predictions
            img_noise_pred_doubled, params_noise_pred_doubled = self.model(
                x_t_doubled, params_t_doubled, t_doubled, objectives_doubled, cfg_mask
            )
            
            # Split conditional and unconditional predictions
            img_noise_cond, img_noise_uncond = img_noise_pred_doubled.chunk(2, dim=0)
            params_noise_cond, params_noise_uncond = params_noise_pred_doubled.chunk(2, dim=0)
            
            # Apply classifier-free guidance
            img_noise_pred = img_noise_uncond + cfg_scale * (img_noise_cond - img_noise_uncond)
            params_noise_pred = params_noise_uncond + cfg_scale * (params_noise_cond - params_noise_uncond)
        else:
            # Standard prediction without CFG
            cfg_mask = torch.ones(x_t.shape[0], dtype=torch.bool, device=x_t.device)
            img_noise_pred, params_noise_pred = self.model(x_t, params_t, t, objectives, cfg_mask)
        
        return img_noise_pred, params_noise_pred

    @torch.no_grad()
    def sample_one_step(self, x_t, params_t, time_step: int, prev_time_step: int, 
                       eta: float, objectives, cfg_scale=1.0):
        """
        DDIM sampling step for both images and parameters
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # Get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)
        
        # For parameters
        params_alpha_t = extract(self.alpha_t_bar, t, params_t.shape)
        params_alpha_t_prev = extract(self.alpha_t_bar, prev_t, params_t.shape)

        # Predict noise
        img_noise_pred, params_noise_pred = self.predict_noise(x_t, params_t, t, objectives, cfg_scale)

        # Calculate sigma for image and parameters
        img_sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        params_sigma_t = eta * torch.sqrt((1 - params_alpha_t_prev) / (1 - params_alpha_t) * 
                                        (1 - params_alpha_t / params_alpha_t_prev))
        
        # Sample random noise for image and parameters
        img_epsilon_t = torch.randn_like(x_t)
        params_epsilon_t = torch.randn_like(params_t)
        
        # Calculate x_{t-1} and params_{t-1} using DDIM formula
        x_t_minus_one = (
            torch.sqrt(alpha_t_prev / alpha_t) * x_t +
            (torch.sqrt(1 - alpha_t_prev - img_sigma_t ** 2) - 
             torch.sqrt((alpha_t_prev * (1 - alpha_t)) / alpha_t)) * img_noise_pred +
            img_sigma_t * img_epsilon_t
        )
        
        params_t_minus_one = (
            torch.sqrt(params_alpha_t_prev / params_alpha_t) * params_t +
            (torch.sqrt(1 - params_alpha_t_prev - params_sigma_t ** 2) - 
             torch.sqrt((params_alpha_t_prev * (1 - params_alpha_t)) / params_alpha_t)) * params_noise_pred +
            params_sigma_t * params_epsilon_t
        )
        
        return x_t_minus_one, params_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, params_t, objectives, cfg_scale=1.0, steps: int = 50, 
                method="linear", eta=0.0, only_return_x_0: bool = True, interval: int = 1):
        """
        Sample from the diffusion model using DDIM.
        
        Parameters:
            x_t: Standard Gaussian noise for images [batch_size, channels, height, width]
            params_t: Standard Gaussian noise for parameters [batch_size, n_params]
            objectives: Target objective values [batch_size, obj_dim]
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            steps: Number of sampling steps
            method: Sampling method ("linear" or "quadratic")
            eta: DDIM vs. DDPM coefficient (0=DDIM, 1=DDPM)
            only_return_x_0: If True, only return final results
            interval: Interval for saving intermediate steps
            
        Returns:
            If only_return_x_0=True: Tuple of (image, parameters)
            If only_return_x_0=False: Tuple of (image_sequence, parameters_sequence)
        """
        # Define time steps based on specified method
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int32)
        else:
            raise NotImplementedError(f"Sampling method {method} is not implemented!")

        # Add one to get the final alpha values right
        time_steps = time_steps + 1
        # Previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        img_samples = [x_t] if not only_return_x_0 else []
        param_samples = [params_t] if not only_return_x_0 else []
        
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps, desc="DDIM Sampling") as pbar:
            for i in pbar:
                x_t, params_t = self.sample_one_step(
                    x_t, params_t, time_steps[i], time_steps_prev[i], eta, objectives, cfg_scale
                )

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    img_samples.append(torch.clip(x_t, -1.0, 1.0))
                    param_samples.append(params_t)

                pbar.set_postfix({
                    "step": i + 1, 
                    "cfg_scale": f"{cfg_scale:.1f}"
                })

        if only_return_x_0:
            return torch.clip(x_t, -1.0, 1.0), params_t
        
        # Stack all samples along a new dimension
        img_sequence = torch.stack(img_samples, dim=1)
        param_sequence = torch.stack(param_samples, dim=1)
        
        return img_sequence, param_sequence


def create_target_objectives(batch_size, target_value, device='cpu'):
    """
    Create target objective tensors for conditional generation.
    
    Args:
        batch_size: Number of samples to generate
        target_value: Target objective value (scalar or list/array)
        device: Device to create tensors on
        
    Returns:
        objectives: [batch_size, obj_dim] tensor
    """
    if isinstance(target_value, (float, int)):
        # Scalar target
        objectives = torch.full((batch_size, 1), target_value, device=device, dtype=torch.float32)
    elif isinstance(target_value, (list, tuple, np.ndarray)):
        # Vector target
        objectives = torch.tensor(target_value, device=device, dtype=torch.float32)
        if objectives.dim() == 1:
            objectives = objectives.unsqueeze(0).expand(batch_size, -1)
        elif objectives.shape[0] == 1:
            objectives = objectives.expand(batch_size, -1)
        elif objectives.shape[0] != batch_size:
            raise ValueError(f"Target value batch size {objectives.shape[0]} doesn't match {batch_size}")
    else:
        raise ValueError(f"Unsupported target_value type: {type(target_value)}")
    
    return objectives