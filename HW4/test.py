import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from utils.schedulers import LinearWarmupCosineAnnealingLR
import os
import torch.nn as nn
from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset, TestSpecificDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR
import lightning.pytorch as pl
import torch.nn.functional as F

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
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=150)
        return [optimizer], [scheduler]

def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))

def test_Derain_Dehaze(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.to(device), clean_patch.to(device)
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))

def test_all_in_one(net, dataset, output_path):
    subprocess.check_output(['mkdir', '-p', output_path])
    testloader = DataLoader(dataset, batch_size=1, pin_memory=False, shuffle=False, num_workers=0)

    with torch.no_grad():
        for ([degraded_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)
            restored = net(degrad_patch)
            save_image_tensor(restored, os.path.join(output_path, degraded_name[0] + '.png'))
    print(f"All-in-one restoration complete. Images saved to {output_path}")

if __name__ == '__main__':
    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--cuda', type=int, default=0, help='CUDA device index (if available)')
    test_parser.add_argument('--mps', type=int, default=1, help='Use MPS (0 for disable, 1 for enable)')
    test_parser.add_argument('--mode', type=int, default=0, help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')
    test_parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    test_parser.add_argument('--test_path', type=str, default="test/degraded/", help='save path of test degraded images (rain, snow, etc.)')
    test_parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    test_parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    test_parser.add_argument('--ckpt_name', type=str, default="epoch=76-step=123200.ckpt", help='checkpoint save path')
    testopt = test_parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    
    # Set device
    if torch.cuda.is_available() and testopt.cuda >= 0:
        device = torch.device(f"cuda:{testopt.cuda}")
        torch.cuda.set_device(testopt.cuda)
    elif torch.backends.mps.is_available() and testopt.mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision('medium')
    ckpt_path = "train_ckpt/" + testopt.ckpt_name

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    # Only initialize denoising datasets for mode 0
    if testopt.mode == 0:
        base_path = testopt.denoise_path
        for i in denoise_splits:
            testopt.denoise_path = os.path.join(base_path, i)
            denoise_testset = DenoiseTestDataset(testopt)
            denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    net = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    net.eval()

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)
            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)
            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)
    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.test_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.test_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze(net, derain_set, task="derain")
    elif testopt.mode == 2:
        print('Start testing SOTS...')
        derain_base_path = testopt.dehaze_path
        name = derain_splits[0]
        testopt.dehaze_path = os.path.join(derain_base_path, name)
        derain_set = DerainDehazeDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    elif testopt.mode == 3:
        print('Start all-in-one testing for derain and desnow...')
        test_set = TestSpecificDataset(testopt.test_path)
        test_all_in_one(net, test_set, testopt.output_path)