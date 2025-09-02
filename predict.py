from utils.tools import load_yaml
from model.UNetPredictor import PropertyPredictionUNet
from utils.engine import PropertyPredictionTrainer
from utils.tools import train_one_epoch, load_yaml
import torch
from argparse import ArgumentParser

from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

import numpy as np
import joblib


def parse_option():
	parser = ArgumentParser()
	parser.add_argument("-cp", "--checkpoint_path", type=str, default="predictor/unet.pth")
	parser.add_argument("-sp_param", "--scaler_path_param", type=str, default="predictor/param_scaler.pkl")
	parser.add_argument("-sp_prop", "--scaler_path_prop", type=str, default="predictor/prop_scaler.pkl")
	
	parser.add_argument("--device", type=str, default="cuda")
	
	# generator param
	parser.add_argument("-bs", "--batch_size", type=int, default=16)
	
	args = parser.parse_args()
	return args

def save_sample_image(images: torch.Tensor, result_only, nrow, **kwargs):
	"""
	concat all image including intermediate process into a picture.
	
	Parameters:
		images: images including intermediate process,
			a tensor with shape (batch_size, sample, channels, height, width).
		**kwargs: other arguments for `torchvision.utils.make_grid`.
	
	Returns:
		concat image, a tensor with shape (height, width, channels).
	"""
	if result_only:
		nrow = nrow
		nrow = images.shape[1]
	else:
		nrow = images.shape[1]
	
	images = images * 0.5 + 0.5 # process [-1,1] to [0,1]
	images = images.squeeze(2) # remove channel, since we are black n white
	images = images.reshape(-1, 1, 128, 128) # concat the images together first
	
	grid = make_grid(images, nrow=nrow, padding=2, normalize=True)
	if grid.shape[0] == 1:  # If grayscale, convert to 3 channels
		grid = grid.repeat(3, 1, 1)
	grid_image = transforms.ToPILImage()(grid)
	
	grid_image.save(args.save_path+"images.png")


@torch.no_grad()
def generate(args):
	device = torch.device(args.device)

	config = load_yaml("config.yml", encoding="utf-8")
	
	cp = torch.load(args.checkpoint_path)
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

	# generate Gaussian noise
	z_t = torch.randn(
		(args.batch_size, cp["config"]["Model"]["in_channels"], *cp["config"]["Dataset"]["image_size"]),
		device=device)
	z_params = torch.randn((args.batch_size, cp["config"]["Dataset"]["param_dim"]), device=device)
	extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
	
	target_objs = create_target_objectives(
		batch_size=args.batch_size, target_value=-1.0, device='cuda')

	images, params = sampler(
		z_t, z_params, only_return_x_0=args.result_only, interval=args.interval, 
		objectives=target_objs, cfg_scale=args.cfg,
		**extra_param)

	param_scaler = joblib.load(args.scaler_path_param)
	params = params.cpu().numpy()
	params = param_scaler.inverse_transform(params)

	save_sample_image(images, result_only=args.result_only, nrow=args.nrow)
	np.savetxt(args.save_path+"params.csv", params, delimiter=",", fmt='%f',
			   header=
			   "diameter, pitch, height, nQWs, growth_Temp_QW,growth_AsP_QW, growth_InP_barrier, growth_time_cap",
			  )

if __name__ == "__main__":
    args = parse_option()
    generate(args)
