import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
import tensorboard
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)
        return [optimizer], [scheduler]

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=False, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    # Dynamically set accelerator and devices
    if torch.cuda.is_available() and opt.num_gpus > 0:
        accelerator = "gpu"
        devices = opt.num_gpus
        strategy = "ddp" if opt.num_gpus > 1 else "auto"
    elif torch.backends.mps.is_available() and opt.mps:  # Check for MPS
        accelerator = "mps"
        devices = 1  # MPS supports single device
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = None
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model=model, train_dataloaders=trainloader)

if __name__ == '__main__':
    main()