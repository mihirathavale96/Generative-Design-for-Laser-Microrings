import torch
from torch.utils.data import DataLoader, Dataset
from pathlib2 import Path, Iterable
from typing import Union, Iterable
import pandas as pd
import numpy as np

import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from PIL import Image  # For compatibility with torchvision.transforms
from skimage.filters import threshold_multiotsu
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
import cv2

from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler


class ImageDataset(Dataset):
	def __init__(self, dataframe):
		
		self.images = [np.array(img, dtype=np.uint8) for img in dataframe['binarized_cropped_optical_image']]
	
		self.transform = v2.Compose([
			v2.RandomHorizontalFlip(p=0.5),
			v2.RandomVerticalFlip(p=0.5),
			v2.ToImage(), 
			v2.ToDtype(torch.float32, scale=True),  # Converts [0, 255] to [0, 1]
			v2.Normalize(mean=[0.5], std=[0.5]),    # Maps [0, 1] → [-1, 1]
		])
	
		self.params = dataframe[['diameter', 'pitch', 'height', 'nQWs', 'growth_Temp_QW',
								 'growth_AsP_QW', 'growth_InP_barrier', 'growth_time_cap']].values
		self.prop = dataframe[['lasing_threshold']].values
		
		self.params_scaler = MinMaxScaler(feature_range=(-1, 1))
		self.prop_scaler = PowerTransformer()
	
	def __len__(self):
		return len(self.images)
	
	def preprocessing(self):
		'''
		for i in range(len(self.images)):
			image = self.images[i]
		
			# Convert grayscale images to RGB by duplicating the single channel
			if len(image.shape) == 2:  # If grayscale (only 1 channel)
				image = np.stack([image] * 3, axis=-1)  # Duplicate the channel to form an RGB image
	
			# Convert to HSV using OpenCV
			image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	
			# Extract the V (value) channel, which is the brightness channel
			value_channel = image[:, :, 2]
	
			# Apply adaptive thresholding
			# Use adaptiveThreshold with a mean value method, block size of 11, and a constant of 5
			image = cv2.adaptiveThreshold(value_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
												cv2.THRESH_BINARY, 13, 6)
	
			# Normalize binary convention (ensure foreground is white and background dark)
			#image = self.normalize_binary(image)
	
			# Convert binary back to a PIL Image for consistency
			image = Image.fromarray(image)
	
			self.images_new.append(image)
		   '''
	
		self.params_norm = self.params_scaler.fit_transform(self.params)
		self.params_norm = torch.tensor(self.params_norm, dtype=torch.float32)
		self.prop_norm = self.prop_scaler.fit_transform(self.prop)
		self.prop_norm = torch.tensor(self.prop_norm, dtype=torch.float32)
	
	def __getitem__(self, idx):
		image = self.images[idx]
		image = self.transform(image)
		image = v2.Resize((128, 128))(image) 
	
		param = self.params_norm[idx]
		prop = self.prop_norm[idx]
		
		return image, param, prop
	

def create_dataset(batch_size, **kwargs):
	path = Path.cwd()
	df= pd.read_pickle(f'{path}/dataset/images_crop_only.pkl')
	
	dataset = ImageDataset(df)
	dataset.preprocessing()
	
	loader_params = dict(
		shuffle=kwargs.get("shuffle", True),
		drop_last=kwargs.get("drop_last", True),
		pin_memory=kwargs.get("pin_memory", True),
		num_workers=kwargs.get("num_workers", 4),
	)
	dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)
	return dataloader, dataset.params_scaler, dataset.prop_scaler
