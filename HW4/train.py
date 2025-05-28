import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
import numpy as np
import wandb
import tensorboard
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from kornia.losses import SSIMLoss

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11)
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        l1_loss = self.l1_loss(restored, clean_patch)
        ssim_loss = self.ssim_loss(restored, clean_patch)
        loss = 0.7 * l1_loss + 0.3 * ssim_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_l1_loss", l1_loss, on_step=True, on_epoch=True)
        self.log("train_ssim_loss", ssim_loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=0,
            threshold=1e-4,
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }



# def main():
#     print("Options")
#     print(opt)
#     if opt.wblogger is not None:
#         logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
#     else:
#         logger = TensorBoardLogger(save_dir = "logs/")

#     trainset = PromptTrainDataset(opt)
#     checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
#     trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
#                              drop_last=True, num_workers=opt.num_workers)
    
#     model = PromptIRModel()
    
#     trainer = pl.Trainer(max_epochs=opt.epochs,
#                          accelerator="gpu",
#                          devices=opt.num_gpus,
#                          strategy="ddp_find_unused_parameters_true",
#                          logger=logger,
#                          callbacks=[checkpoint_callback])
#     trainer.fit(model=model, train_dataloaders=trainloader)

def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                            drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    # Dynamically set accelerator and devices
    if torch.cuda.is_available() and opt.num_gpus > 0:
        accelerator = "gpu"
        devices = opt.num_gpus
        strategy = "ddp" if opt.num_gpus > 1 else "auto"  # Use DDP only for multi-GPU, auto for single-GPU
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



