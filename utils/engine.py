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
	def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int, img_weight: float = 1.0, param_weight: float = 1.0):
		super().__init__()
		self.model = model
		self.T = T
		self.img_weight = img_weight  # Weight for image loss
		self.param_weight = param_weight  # Weight for parameter loss
	
		# generate T steps of beta
		self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
	
		# calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
		alpha_t = 1.0 - self.beta_t
		alpha_t_bar = torch.cumprod(alpha_t, dim=0)
	
		# calculate and store two coefficient of $q(x_t | x_0)$
		self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
		self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))
	
	def forward(self, x_0, params_0):
		"""
		Train the diffusion model to denoise both image and parameters.
		
		Args:
			x_0: Clean images [batch, 1, 128, 128]
			params_0: Clean parameters [batch, n_params]
		"""
		batch_size = x_0.shape[0]
		
		# Get a random training step t ~ Uniform({1, ..., T})
		t = torch.randint(self.T, size=(batch_size,), device=x_0.device)
		
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
		
		# Predict noise of image, params and prop
		img_pred, params_pred = self.model(x_t, t, params_t)
		
		# Calculate loss for image, parameters and prop
		img_loss = F.mse_loss(img_pred, img_noise, reduction="none")
		img_loss = torch.sum(img_loss) * self.img_weight
		
		params_loss = F.mse_loss(params_pred, params_noise, reduction="none")
		params_loss = torch.sum(params_loss) * self.param_weight

		# Total loss
		total_loss = img_loss + params_loss
		
		return total_loss

class EnergyTrainer(nn.Module):
	def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int,):
		super().__init__()
		self.model = model
		self.T = T
	
		# generate T steps of beta
		self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
	
		# calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
		alpha_t = 1.0 - self.beta_t
		alpha_t_bar = torch.cumprod(alpha_t, dim=0)
	
		# calculate and store two coefficient of $q(x_t | x_0)$
		self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar))
		self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar))
	
	def forward(self, x_0, params_0, prop_0):
		"""
		Train the diffusion model to predict time-dependent property (energy).
		
		Args:
			x_0: Clean images [batch, 1, 128, 128]
			params_0: Clean parameters [batch, n_params]
			prop_0: Clean property value [batch, n_props]
		"""
		batch_size = x_0.shape[0]
		
		# Get a random training step t ~ Uniform({1, ..., T})
		t = torch.randint(self.T, size=(batch_size,), device=x_0.device)
		
		# Generate noise for both image and parameters
		img_noise = torch.randn_like(x_0)
		params_noise = torch.randn_like(params_0)
		prop_noise = torch.randn_like(prop_0)
		
		# Add noise to the image at timestep t
		x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
			   extract(self.noise_rate, t, x_0.shape) * img_noise)
		
		# Add noise to the parameters at timestep t
		params_signal_rate = extract(self.signal_rate, t, params_0.shape)
		params_noise_rate = extract(self.noise_rate, t, params_0.shape)
		params_t = params_signal_rate * params_0 + params_noise_rate * params_noise
	
		# Add noise to the property at timestep t
		prop_signal_rate = extract(self.signal_rate, t, prop_0.shape)
		prop_noise_rate = extract(self.noise_rate, t, prop_0.shape)
		prop_t = prop_signal_rate * prop_0 + prop_noise_rate * prop_noise
		
		# Predict noise of property
		prop_pred = self.model(x_t, t, params_t)
		
		prop_loss = F.mse_loss(prop_pred, prop_t, reduction="none")
		prop_loss = torch.sum(prop_loss)
		
		return prop_loss


class DDPMSampler(nn.Module):
	def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
		super().__init__()
		self.model = model
		self.T = T
	
		# generate T steps of beta
		self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
	
		# calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
		alpha_t = 1.0 - self.beta_t
		alpha_t_bar = torch.cumprod(alpha_t, dim=0)
		alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)
	
		self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
		self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
		self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))
	
	@torch.no_grad()
	def cal_mean_variance(self, x_t, params_t, t):
		"""
		Calculate the mean and variance for both image and parameters
		"""
		# Predict noise for both image and parameters
		img_noise_pred, params_noise_pred, prop_noise_pred = self.model(x_t, t, params_t)
		
		# Calculate mean and variance for image
		img_mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * img_noise_pred
		img_var = extract(self.posterior_variance, t, x_t.shape)
		
		# Calculate mean and variance for parameters
		params_mean = extract(self.coeff_1, t, params_t.shape) * params_t - extract(self.coeff_2, t, params_t.shape) * params_noise_pred
		params_var = extract(self.posterior_variance, t, params_t.shape)
		
		return img_mean, img_var, params_mean, params_var
	
	@torch.no_grad()
	def sample_one_step(self, x_t, params_t, time_step: int):

		"""
		Calculate x_{t-1} and params_{t-1} according to x_t and params_t
		"""
		t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
		img_mean, img_var, params_mean, params_var = self.cal_mean_variance(x_t, params_t, t)
	
		# Sample noise for both image and parameters if time_step > 0
		img_z = torch.randn_like(x_t) if time_step > 0 else 0
		params_z = torch.randn_like(params_t) if time_step > 0 else 0
		
		# Calculate x_{t-1} and params_{t-1}
		x_t_minus_one = img_mean + torch.sqrt(img_var) * img_z
		params_t_minus_one = params_mean + torch.sqrt(params_var) * params_z
	
		# Check for NaN values
		if torch.isnan(x_t_minus_one).int().sum() != 0 or torch.isnan(params_t_minus_one).int().sum() != 0:
			raise ValueError("nan in tensor!")
	
		return x_t_minus_one, params_t_minus_one
	
	@torch.no_grad()
	def forward(self, x_t, params_t, only_return_x_0: bool = True, interval: int = 1, **kwargs):
		"""
		Sample from the diffusion model.
		
		Parameters:
			x_t: Standard Gaussian noise for images [batch_size, channels, height, width]
			params_t: Standard Gaussian noise for parameters [batch_size, n_params]
			only_return_x_0: If True, only return final results. If False, return intermediate steps.
			interval: Interval for saving intermediate steps (when only_return_x_0=False)
			
		Returns:
			If only_return_x_0=True: Tuple of (image, parameters)
			If only_return_x_0=False: Tuple of (image_sequence, parameters_sequence)
		"""
		img_samples = [x_t]
		param_samples = [params_t]
		
		with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
			for time_step in sampling_steps:
				x_t, params_t = self.sample_one_step(x_t, params_t, time_step)
	
				if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
					img_samples.append(torch.clip(x_t, -1.0, 1.0))
					param_samples.append(params_t)  # No clipping for parameters
	
				sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(img_samples)})
	
		if only_return_x_0:
			return x_t, params_t  # Return final image and parameters
		
		# Stack all samples along a new dimension
		img_sequence = torch.stack(img_samples, dim=1)  # [batch_size, samples, channels, height, width]
		param_sequence = torch.stack(param_samples, dim=1)  # [batch_size, samples, n_params]
		
		return img_sequence, param_sequence


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, params_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # Get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)
        
        # For parameters
        params_alpha_t = extract(self.alpha_t_bar, t, params_t.shape)
        params_alpha_t_prev = extract(self.alpha_t_bar, prev_t, params_t.shape)

        # Predict noise using model for both image and parameters
        img_noise_pred, params_noise_pred = self.model(x_t, t, params_t)

        # Calculate sigma for image and parameters
        img_sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        params_sigma_t = eta * torch.sqrt((1 - params_alpha_t_prev) / (1 - params_alpha_t) * (1 - params_alpha_t / params_alpha_t_prev))
        
        # Sample random noise for image and parameters
        img_epsilon_t = torch.randn_like(x_t)
        params_epsilon_t = torch.randn_like(params_t)
        
        # Calculate x_{t-1} and params_{t-1}
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - img_sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * img_noise_pred +
                img_sigma_t * img_epsilon_t
        )
        
        params_t_minus_one = (
                torch.sqrt(params_alpha_t_prev / params_alpha_t) * params_t +
                (torch.sqrt(1 - params_alpha_t_prev - params_sigma_t ** 2) - torch.sqrt(
                    (params_alpha_t_prev * (1 - params_alpha_t)) / params_alpha_t)) * params_noise_pred +
                params_sigma_t * params_epsilon_t
        )
        
        return x_t_minus_one, params_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, params_t, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        Sample from the diffusion model using DDIM.
        
        Parameters:
            x_t: Standard Gaussian noise for images [batch_size, channels, height, width]
            params_t: Standard Gaussian noise for parameters [batch_size, n_params]
            steps: Number of sampling steps
            method: Sampling method ("linear" or "quadratic")
            eta: DDIM vs. DDPM coefficient (0=DDIM, 1=DDPM)
            only_return_x_0: If True, only return final results. If False, return intermediate steps.
            interval: Interval for saving intermediate steps (when only_return_x_0=False)
            
        Returns:
            If only_return_x_0=True: Tuple of (image, parameters)
            If only_return_x_0=False: Tuple of (image_sequence, parameters_sequence)
        """
        # Define time steps based on specified method
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # Add one to get the final alpha values right
        time_steps = time_steps + 1
        # Previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        img_samples = [x_t]
        param_samples = [params_t]
        
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t, params_t = self.sample_one_step(x_t, params_t, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    img_samples.append(torch.clip(x_t, -1.0, 1.0))
                    param_samples.append(params_t)  # No clipping for parameters

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(img_samples)})

        if only_return_x_0:
            return x_t, params_t  # Return final image and parameters
        
        # Stack all samples along a new dimension
        img_sequence = torch.stack(img_samples, dim=1)  # [batch_size, samples, channels, height, width]
        param_sequence = torch.stack(param_samples, dim=1)  # [batch_size, samples, n_params]
        
        return img_sequence, param_sequence