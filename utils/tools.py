from typing import Optional, Union
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision import transforms
from PIL import Image
from pathlib2 import Path
import yaml


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg

def train_one_epoch(trainer, loader, optimizer, device, epoch):
	"""
	Train for one epoch with modified data loader that returns images, parameters and property.
	
	Args:
		trainer: GaussianDiffusionTrainer instance
		loader: data loader that yields (images, parameters, property) tuples
		optimizer: optimizer
		device: device to use
		epoch: current epoch number
	"""
	trainer.train()
	total_loss, total_num = 0., 0
	
	with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
		for batch in data:
			optimizer.zero_grad()
			
			images, params, prop = batch
			
			x_0 = images.to(device)
			params_0 = params.to(device)
			prop_0 = prop.to(device)
			
			# Forward pass with both image and parameters
			loss = trainer(x_0, params_0)
			
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			total_num += x_0.shape[0]
			
			data.set_description(f"Epoch: {epoch}")
			data.set_postfix(ordered_dict={
				"train_loss": total_loss / total_num,
			})
	
	return total_loss / total_num

def train_one_epoch_energy(trainer, loader, optimizer, device, epoch):
	"""
	Train for one epoch with modified data loader that returns images, parameters and property.
	
	Args:
		trainer: GaussianDiffusionTrainer instance
		loader: data loader that yields (images, parameters, property) tuples
		optimizer: optimizer
		device: device to use
		epoch: current epoch number
	"""
	trainer.train()
	total_loss, total_num = 0., 0
	
	with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
		for batch in data:
			optimizer.zero_grad()
			
			images, params, prop = batch
			
			x_0 = images.to(device)
			params_0 = params.to(device)
			prop_0 = prop.to(device)
			
			# Forward pass with image, params, prop
			loss = trainer(x_0, params_0, prop_0)
			
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			total_num += x_0.shape[0]
			
			data.set_description(f"Epoch: {epoch}")
			data.set_postfix(ordered_dict={
				"train_loss": total_loss / total_num,
			})
	
	return total_loss / total_num


