import random
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.core import LightningModule

from inception import InceptionV3
import fid

def DCGenerator(z_sz, gf_sz):
    gf_sz = gf_sz * 8
    tconv = partial(nn.ConvTranspose2d, kernel_size=4, stride=2, padding=1, bias=False)
    layers = [tconv(z_sz, gf_sz, stride=1, padding=0)]
    for _ in range(2):
        layers.extend([nn.BatchNorm2d(gf_sz),
                       nn.LeakyReLU(0.2, inplace=True),
                       tconv(gf_sz, gf_sz // 2)])
        gf_sz = gf_sz // 2
    return nn.Sequential(*layers, nn.Conv2d(gf_sz, 3, 1), nn.Tanh())

def DCDiscriminator(df_sz):
    conv = partial(nn.Conv2d, kernel_size=4, stride=2, padding=1, bias=False)
    layers = [conv(3, df_sz, kernel_size=5, stride=1, padding=2), nn.LeakyReLU(0.2, inplace=True)]
    for _ in range(2):
        layers.extend([conv(df_sz, df_sz*2),
                       nn.BatchNorm2d(df_sz*2),
                       nn.LeakyReLU(0.2, inplace=True)])
        df_sz *= 2
    return nn.Sequential(*layers, conv(df_sz, 1, stride=1, padding=0), nn.Flatten())

class WGanGP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.generator = DCGenerator(hparams.z_sz, hparams.gf_sz)
        self.discriminator = DCDiscriminator(hparams.df_sz)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[hparams.fid_dims]
        self.inception = InceptionV3([block_idx]).eval()

        self.constant_z = nn.Parameter(torch.randn(64, hparams.z_sz, 1, 1))

    def forward(self, x):
        return self.generator(x)

    def randn_z(self, imgs):
        return torch.randn(imgs.size(0), self.hparams.z_sz, 1, 1, device=imgs.device)

    def training_step_generator(self, imgs):
        generated_images = self(self.randn_z(imgs))
        g_loss = torch.relu(1 - self.discriminator(generated_images)).mean()

        tqdm_dict = {'g_loss': g_loss}
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def training_step_discriminator(self, imgs):
        with torch.no_grad():
            generated_images = self(self.randn_z(imgs)).detach()
        fake_loss = torch.relu(1 + self.discriminator(generated_images)).mean()
        real_loss = torch.relu(1 - self.discriminator(imgs)).mean()

        d_loss = (fake_loss + real_loss) / 2
        gp_loss = self.calc_gradient_penalty(imgs, generated_images) * self.hparams.gp_coef
        tqdm_dict = {'d_loss': d_loss, 'gp_loss':gp_loss}

        output = OrderedDict({
            'loss': d_loss + gp_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def get_interpolation(self, imgs, generated_images):
        alpha = torch.rand(imgs.size(0), 1, 1, 1, device=imgs.device)
        return imgs * alpha + generated_images * (1 - alpha)

    def calc_gradient_penalty(self, imgs, generated_images):
        interpolation = self.get_interpolation(imgs, generated_images)
        interpolation.requires_grad=True
        d_out = self.discriminator(interpolation)
        grads = torch.autograd.grad(d_out, interpolation, grad_outputs=torch.ones_like(d_out), create_graph=True)[0]
        grads = torch.norm(grads, p=2, dim=[1,2,3]) - 1
        return (grads ** 2).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        if optimizer_idx == 0:
            return self.training_step_generator(imgs)
        if optimizer_idx == 1:
            return self.training_step_discriminator(imgs)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        def incept_act(x_):
            return F.adaptive_avg_pool2d(self.inception(x_)[0], 1).view(x.size(0), 1, -1)
        with torch.no_grad():
            return torch.cat([incept_act(x),
                              incept_act(self(self.randn_z(x)))], dim=1)

    def validation_epoch_end(self, outputs):
        def get_mu_sig(x):
            return np.mean(x, axis=0), np.cov(x, rowvar=False)
        mu1, sig1 = get_mu_sig(torch.cat([x[:,0].cpu() for x in outputs]).numpy())
        mu2, sig2 = get_mu_sig(torch.cat([x[:,1].cpu() for x in outputs]).numpy())

        fid_ = fid.calculate_frechet_distance(mu1, sig1, mu2, sig2)
        return OrderedDict({
                 'val_loss' : fid_,
                 'log' : {'fid_loss' : fid_}
               })

    def configure_optimizers(self):
        lr_g = self.hparams.lr_g
        lr_d = self.hparams.lr_d
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    def __get_dataloader(self, download):
        transform = transforms.Compose([torchvision.transforms.CenterCrop(178),
                                        torchvision.transforms.Resize(16),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5]*3, [0.5]*3)])
        dataset = CelebA(self.hparams.data_root, split='all', download=download, transform=transform)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=32)

    def prepare_data(self):
        _ = self.__get_dataloader(True)

    def train_dataloader(self):
        return self.__get_dataloader(False)

    def val_dataloader(self):
        return self.__get_dataloader(False)

    def on_epoch_end(self):
        with torch.no_grad():
            grid = torchvision.utils.make_grid((self(self.constant_z) + 1) / 2).cpu()
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--gf-sz', default=64, type=int)
        parser.add_argument('--df-sz', default=64, type=int)
        parser.add_argument('--z-sz', default=100, type=int)
        parser.add_argument('--lr-g', default=1e-4, type=float)
        parser.add_argument('--lr-d', default=4e-4, type=float)
        parser.add_argument('--b1', default=0.5, type=float)
        parser.add_argument('--b2', default=0.999, type=float)
        parser.add_argument('--gp-coef', default=10.0, type=float)

        parser.add_argument('--fid-dims', default=2048, type=int,
                               help='Dimensionality of features from InceptionNetV3. '
                               '[64: first maxpool | 192: second maxpool | 768: pre-aux | 2048: final average pooling]')

        parser.add_argument('--data-root', default='./', type=str)
        return parser

def get_args():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--exp-name', type=str,
                               help='experiment name')
    parent_parser.add_argument('--epochs', default=200, type=int,
                               help='number of epochs to run')
    parent_parser.add_argument('--gpus', type=str, default="-1",
                               help='String of gpus to train on seperated with \',\' as delimiter (or -1 for all available)')
    parent_parser.add_argument('--batch-size', default=64, type=int,
                               help='batch size to use')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('--seed', default=-1, type=int,
                               help='seed for deterministic routine (or -1 for random rutine)')

    parser = WGanGP.add_model_specific_args(parent_parser)
    return parser.parse_args()

def main(hparams):
    if hparams.seed != -1:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)

    logger = TensorBoardLogger("tb_logs", name=hparams.exp_name)
    gpus = list(map(int, hparams.gpus.split(',')))
    if len(gpus)==1 and gpus[0] == -1:
        gpus = -1

    trainer = pl.Trainer(
        early_stop_callback=False,
        logger=logger,
        gpus=gpus,
        max_epochs=hparams.epochs,
        val_percent_check=0.05,
        use_amp=hparams.use_16bit,
    )

    model = WGanGP(hparams)
    trainer.fit(model)

if __name__ == '__main__':
    main(get_args())
