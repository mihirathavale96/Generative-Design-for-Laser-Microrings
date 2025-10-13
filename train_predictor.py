from pathlib import Path
import sys
sys.path.append('.')

from dataset import create_dataset
from model.UNetPredictor import UNetPredictor
from utils.engine import PropertyPredictionTrainer
from utils.tools import train_one_epoch, load_yaml

import torch
from utils.callbacks import ModelCheckpoint

import joblib
import argparse


def train(config):
    # Create predictor checkpoint directory if it doesn't exist
    Path(config['CallbackPredictor']['filepath']).parent.mkdir(parents=True, exist_ok=True)
    
    consume = config["consume"]
    if consume:
        cp = torch.load(config["consume_path"])
        config = cp["config"]
    print(config)
    
    device = torch.device(config["device"])
    loader, param_scaler, prop_scaler = create_dataset(**config["Dataset"])
    start_epoch = 1

    joblib.dump(param_scaler, 'predictor/param_scaler.pkl')
    joblib.dump(prop_scaler, 'predictor/prop_scaler.pkl')
    
    model = UNetPredictor(**config["ModelPredictor"]).to(device)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the predictor model with required dataset path.")
    parser.add_argument('--datafilepath', type=str, required=True, help="Path to dataset file (e.g., /content/drive/MyDrive/... or /scratch/your_user/...)")
    parser.add_argument('--predictor_checkpoint_dir', type=str, default='./predictor', help="Directory for predictor checkpoints (e.g., /content/drive/MyDrive/predictor or /scratch/your_user/predictor)")
    args = parser.parse_args()

    config = load_yaml("config.yml", encoding="utf-8")
    
    # Override config with CLI args
    config['Dataset']['datafilepath'] = args.datafilepath
    config['CallbackPredictor']['filepath'] = f"{args.predictor_checkpoint_dir}/unet.pth"
    
    train(config)
