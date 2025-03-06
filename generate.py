from utils.tools import load_yaml
from utils.engine import DDPMSampler, DDIMSampler
from model.UNet import UNet
import torch
from argparse import ArgumentParser

from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image


def parse_option():
	parser = ArgumentParser()
	parser.add_argument("-cp_uncond", "--checkpoint_path_uncond", type=str, default="checkpoint/ddim_uncond.pth")
	parser.add_argument("-cp_energy", "--checkpoint_path_energy", type=str, default="checkpoint/ddim_energy.pth")
	parser.add_argument("-sp_param", "--scaler_path_param", type=str, default="checkpoint/param_scaler.pth")
	parser.add_argument("-sp_prop", "--scaler_path_prop", type=str, default="checkpoint/prop_scaler.pth")
	
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
	
	# generator param
	parser.add_argument("-bs", "--batch_size", type=int, default=16)
	
	# sampler param
	parser.add_argument("--result_only", default=True, action="store_true")
	parser.add_argument("--interval", type=int, default=50)
	
	# DDIM sampler param
	parser.add_argument("--eta", type=float, default=0.0)
	parser.add_argument("--steps", type=int, default=100)
	parser.add_argument("--method", type=str, default="linear", choices=["linear", "quadratic"])
	
	# save image param
	parser.add_argument("--nrow", type=int, default=4)
	parser.add_argument("--show", default=False, action="store_true")
	parser.add_argument("-sp", "--image_save_path", type=str, default=None)
	parser.add_argument("--to_grayscale", default=False, action="store_true")
	
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
	else:
		nrow = images.shape[1]
	
	images = images * 0.5 + 0.5 # process [-1,1] to [0,1]
	images = images.squeeze(2) # remove channel, since we are black n white
	images = images.reshape(-1, 1, 128, 128) # concat the images together first
	
	grid = make_grid(images, nrow=nrow, padding=2, normalize=True)
	if grid.shape[0] == 1:  # If grayscale, convert to 3 channels
		grid = grid.repeat(3, 1, 1)
	grid_image = transforms.ToPILImage()(grid)
	
	grid_image.save("samples.png")


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
	
	# generate Gaussian noise
	z_t = torch.randn((args.batch_size, cp["config"]["Model"]["in_channels"],
					   *cp["config"]["Dataset"]["image_size"]), device=device)

	z_params = torch.randn((args.batch_size, cp["config"]["Dataset"]["param_dim"]), device=device)
	
	extra_param = dict(steps=args.steps, eta=args.eta, method=args.method)

	images, params = sampler(z_t, z_params, only_return_x_0=args.result_only, interval=args.interval, **extra_param)
	
	save_sample_image(images, result_only=args.result_only, nrow=args.nrow)
	

if __name__ == "__main__":
    args = parse_option()
    generate(args)
