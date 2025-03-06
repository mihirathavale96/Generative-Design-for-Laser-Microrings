import numpy as np
import torch
import pandas as pd

from pathlib import Path
import joblib
from argparse import ArgumentParser

from utils.tools import load_yaml
from utils.engine import DDPMSampler, DDIMSampler
from model.UNet import UNet

from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

from torchmetrics.image.fid import FrechetInceptionDistance

def parse_option():
	parser = ArgumentParser()
	parser.add_argument("-cp_uncond", "--checkpoint_path_uncond", type=str, default="checkpoint/ddim_uncond.pth")
	parser.add_argument("-cp_energy", "--checkpoint_path_energy", type=str, default="checkpoint/ddim_energy.pth")
	parser.add_argument("-sp_param", "--scaler_path_param", type=str, default="checkpoint/param_scaler.pth")
	parser.add_argument("-sp_prop", "--scaler_path_prop", type=str, default="checkpoint/prop_scaler.pth")
	parser.add_argument("-dp", "--dataset_path", type=str, default="dataset/images_crop_only.pkl")
	
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
	
	# generator param
	parser.add_argument("-bs", "--batch_size", type=int, default=16)
	parser.add_argument("-nb", "--num_batch", type=int, default=3)
	
	# sampler param
	parser.add_argument("--interval", type=int, default=50)
	
	# DDIM sampler param
	parser.add_argument("--eta", type=float, default=0.0)
	parser.add_argument("--steps", type=int, default=100)
	parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])
	
	args = parser.parse_args()
	return args


@torch.no_grad()
def generate(args):
	device = torch.device(args.device)
	
	config = load_yaml("config.yml", encoding="utf-8")
	
	cp = torch.load(args.checkpoint_path_uncond)
	# load trained model
	model = UNet(**config["Model"]).to(device)
	model.load_state_dict(cp["model"])
	model = model.eval()
	
	if args.sampler == "ddim":
		sampler = DDIMSampler(model, **cp["config"]["Trainer"]).to(device)
	elif args.sampler == "ddpm":
		sampler = DDPMSampler(model, **cp["config"]["Trainer"]).to(device)
	else:
		raise ValueError(f"Unknown sampler: {args.sampler}")	
	
	extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
	
	generated_images = []
	#generated_params = []
	
	for _ in range(args.num_batch):
		z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
						   *cp["config"]["Dataset"]["image_size"]), device=device)
	
		z_params = torch.randn((args.batch_size, cp["config"]["Dataset"]["param_dim"]), device=device)
		
		images, params = sampler(z_t, z_params, only_return_x_0=True, interval=args.interval, **extra_param)
		
		# Process images to [0, 255] uint8
		images = (images * 0.5 + 0.5).clamp(0, 1)
		images = (images * 255).type(torch.uint8).cpu().numpy()
		generated_images.append(images.squeeze(1))  # Remove channel dim
		
		# Inverse transform parameters
		#params = params.cpu().numpy()
		#params_original = params_scaler.inverse_transform(params)
		#generated_params.append(params_original)
	
	generated_images = np.concatenate(generated_images)
	#generated_params = np.concatenate(generated_params)
	
	###################
	# Load real data
	df = pd.read_pickle(args.dataset_path)
	real_images = np.stack([np.array(img) for img in df['binarized_cropped_optical_image']])
	#real_params = df[['diameter', 'pitch', 'height', 'nQWs', 'growth_Temp_QW',
	#                  'growth_AsP_QW', 'growth_InP_barrier', 'growth_time_cap']].values
	
	###################
	fid = FrechetInceptionDistance(normalize=True).to(device)
	
	# Convert to 3-channel and tensor [N, C, H, W]
	real_rgb = np.repeat(real_images[:, None, :, :], 3, axis=1)
	gen_rgb = np.repeat(generated_images[:, None, :, :], 3, axis=1)
	
	real_tensor = torch.tensor(real_rgb, device=device)
	gen_tensor = torch.tensor(gen_rgb, device=device)
	
	fid.update(real_tensor, real=True)
	fid.update(gen_tensor, real=False)
	fid_score = fid.compute().item()
	print(f"{fid_score:.2f}")

if __name__ == "__main__":
    args = parse_option()
    generate(args)
