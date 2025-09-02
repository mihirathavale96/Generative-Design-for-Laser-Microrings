from pathlib import Path
import sys
sys.path.append('.')

from dataset import create_dataset
from model.UNetPredictor import PropertyPredictionUNet
from utils.engine import PropertyPredictionTrainer
from utils.tools import train_one_epoch, load_yaml

import torch
from utils.callbacks import ModelCheckpoint

import joblib


def train(config):
	consume = config["consume"]
	if consume:
		cp = torch.load(config["consume_path"])
		config = cp["config"]
	print(config)
	
	device = torch.device(config["device"])
	loader, param_scaler, prop_scaler = create_dataset(**config["Dataset"])
	start_epoch = 1
	
	model = PropertyPredictionUNet(**config["ModelPredictor"]).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
	trainer = PropertyPredictionTrainer(model).to(device)
	
	model_checkpoint = ModelCheckpoint(**config["CallbackPredictor"])
	
	if consume:
		model.load_state_dict(cp["model"])
		optimizer.load_state_dict(cp["optimizer"])
		model_checkpoint.load_state_dict(cp["model_checkpoint"])
		start_epoch = cp["start_epoch"] + 1
	
	for epoch in range(start_epoch, config["predictor_epochs"] + 1):
		loss = train_one_epoch(trainer, loader, optimizer, device, epoch)
		model_checkpoint.step(loss, model=model.state_dict(), config=config,
							  optimizer=optimizer.state_dict(), start_epoch=epoch,
							  model_checkpoint=model_checkpoint.state_dict())
	
	joblib.dump(param_scaler, 'predictor/param_scaler.pkl')
	joblib.dump(prop_scaler, 'predictor/prop_scaler.pkl')


if __name__ == "__main__":
    config = load_yaml("config.yml", encoding="utf-8")
    train(config)