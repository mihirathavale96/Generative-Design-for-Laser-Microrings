from utils.tools import load_yaml
from utils.engine import DDPMSampler, DDIMSampler, create_target_objectives
from model.UNet import UNet
import torch
from argparse import ArgumentParser

from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image

import numpy as np
import joblib
import datetime


def parse_option():
	parser = ArgumentParser()
	parser.add_argument("-cp", "--checkpoint_path", type=str, default="checkpoint/unet.pth")
	parser.add_argument("-sp_param", "--scaler_path_param", type=str, default="checkpoint/param_scaler.pkl")
	parser.add_argument("-sp_prop", "--scaler_path_prop", type=str, default="checkpoint/prop_scaler.pkl")
	
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
	
	# generator param
	parser.add_argument("-bs", "--batch_size", type=int, default=1024)
	parser.add_argument("-mbs", "--max_batch_size", type=int, default=128)

	# DDIM sampler param
	parser.add_argument("--eta", type=float, default=0.0)
	parser.add_argument("--steps", type=int, default=100)
	parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])

	parser.add_argument("--target", type=int, default=3.6)
	parser.add_argument("--cfg", type=int, default=5.0)

	# save param
	# these determine if you want a progress of the images
	parser.add_argument("--result_only", default=False, action="store_true")
	parser.add_argument("--interval", type=int, default=50)
	
	parser.add_argument("--nrow", type=int, default=4)
	parser.add_argument("-sp", "--save_path", type=str, default="results/")
	
	args = parser.parse_args()
	return args

def save_sample_image(images: torch.Tensor, result_only, nrow, label, **kwargs):
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
		#nrow = nrow
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
	
	grid_image.save(args.save_path+"images_"+label+".png")


@torch.no_grad()
def generate(args):

	#### loading all the stuff ####
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


	#### actual generation ####
	batch_ref_vec = torch.rand(args.batch_size)
	
	if args.batch_size > args.max_batch_size:
		batch_ref_vec_chunk = torch.split(batch_ref_vec, args.max_batch_size, dim=0)

		images, params = [], []

		for chunk in batch_ref_vec_chunk:
			vec_length = len(chunk)

			z_t = torch.randn(
				(vec_length, cp["config"]["Model"]["in_channels"], *cp["config"]["Dataset"]["image_size"]),
				device=device)
			z_params = torch.randn((vec_length, cp["config"]["Dataset"]["param_dim"]), device=device)
			extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
			
			target_objs = create_target_objectives(
				batch_size=vec_length, target_value=args.target,
				obj_dim=cp["config"]["Model"]["obj_dim"], device='cuda')

			images_chunk, params_chunk = sampler(
				z_t, z_params, only_return_x_0=args.result_only, interval=args.interval, 
				objectives=target_objs, cfg_scale=args.cfg,
				**extra_param)

			images.append(images_chunk)
			params.append(params_chunk)

		images = torch.cat(images, dim=0) 
		params = torch.cat(params, dim=0)

	else:
		z_t = torch.randn(
			(args.batch_size, cp["config"]["Model"]["in_channels"], *cp["config"]["Dataset"]["image_size"]),
			device=device)
		z_params = torch.randn((args.batch_size, cp["config"]["Dataset"]["param_dim"]),
							   obj_dim=cp["config"]["Model"]["obj_dim"], device=device)
		extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)
		target_objs = create_target_objectives(
			batch_size=args.batch_size, target_value=args.target, device='cuda')

		images, params = sampler(
			z_t, z_params, only_return_x_0=args.result_only, interval=args.interval, 
			objectives=target_objs, cfg_scale=args.cfg,
			**extra_param)

	#print(images.shape)
	#print(params.shape)


	#### saving everything ####
	param_scaler = joblib.load(args.scaler_path_param)

	if not args.result_only:
		params = params[:, -1, :]
	params = param_scaler.inverse_transform(params.cpu().numpy())

	label = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	save_sample_image(images, result_only=args.result_only, nrow=args.nrow, label=label)
	torch.save(images[:, -1, :, :, :], args.save_path+"images_raw_"+label)
	np.savetxt(args.save_path+"params_"+label+".csv", params, delimiter=",", fmt='%f',
			   header=
			   "diameter, pitch, height, nQWs, growth_Temp_QW,growth_AsP_QW, growth_InP_barrier, growth_time_cap",
			  )

if __name__ == "__main__":
    args = parse_option()
    generate(args)
