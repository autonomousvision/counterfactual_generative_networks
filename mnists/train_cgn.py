import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

import repackage
repackage.up()

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from mnists.config import get_cfg_defaults
from mnists.dataloader import get_dataloaders
from mnists.models import CGN, DiscLin, DiscConv
from utils import save_cfg, load_cfg, children, hook_outputs, Optimizers
from shared.losses import BinaryLoss, PerceptualLoss

def save(x, path, n_row, sz=64):
    x = F.interpolate(x, (sz, sz))
    save_image(x.data, path, nrow=n_row, normalize=True, padding=2)

def sample_image(cgn, sample_path, batches_done, device, n_row=3, n_classes=10):
    """Saves a grid of generated digits"""
    y_gen = np.arange(n_classes).repeat(n_row)
    y_gen = torch.LongTensor(y_gen).to(device)
    mask, foreground, background = cgn(y_gen)
    x_gen = mask * foreground + (1 - mask) * background

    save(x_gen.data, f"{sample_path}/0_{batches_done:d}_x_gen.png", n_row)
    save(mask.data, f"{sample_path}/1_{batches_done:d}_mask.png", n_row)
    save(foreground.data, f"{sample_path}/2_{batches_done:d}_foreground.png", n_row)
    save(background.data, f"{sample_path}/3_{batches_done:d}_background.png", n_row)

def fit(cfg, cgn, discriminator, dataloader, opts, losses, device):

    # directories for experiments
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_path = Path('.') / 'mnists' / 'experiments'
    model_path /= f'cgn_{cfg.TRAIN.DATASET}_{time_str}_{cfg.MODEL_NAME}'
    weights_path = model_path / 'weights'
    sample_path = model_path / 'samples'
    weights_path.mkdir(parents=True, exist_ok=True)
    sample_path.mkdir(parents=True, exist_ok=True)

    # dump config
    save_cfg(cfg, model_path / "cfg.yaml")

    # Training Loop
    L_perc, L_adv, L_binary = losses

    pbar = tqdm(range(cfg.TRAIN.EPOCHS))
    for epoch in pbar:
        for i, data in enumerate(dataloader):

            # Data and adversarial ground truths to device
            x_gt = data['ims'].to(device)
            y_gt = data['labels'].to(device)
            valid = torch.ones(len(y_gt),).to(device)
            fake = torch.zeros(len(y_gt),).to(device)

            #
            #  Train Generator
            #
            opts.zero_grad(['generator'])

            # Sample noise and labels as generator input
            y_gen = torch.randint(cfg.MODEL.N_CLASSES, (len(y_gt),)).to(device)

            # Generate a batch of images
            mask, foreground, background = cgn(y_gen)
            x_gen = mask * foreground + (1 - mask) * background

            # Calc Losses
            validity = discriminator(x_gen, y_gen)

            losses_g = {}
            losses_g['adv'] = L_adv(validity, valid)
            losses_g['binary'] = L_binary(mask)
            losses_g['perc'] = L_perc(x_gen, x_gt)

            # Backprop and step
            loss_g = sum(losses_g.values())
            loss_g.backward()
            opts.step(['generator'], False)

            #
            # Train Discriminator
            #
            opts.zero_grad(['discriminator'])

            # Discriminate real and fake
            validity_real = discriminator(x_gt, y_gt)
            validity_fake = discriminator(x_gen.detach(), y_gen)

            # Losses
            losses_d = {}
            losses_d['real'] = L_adv(validity_real, valid)
            losses_d['fake'] = L_adv(validity_fake, fake)
            loss_d = sum(losses_d.values()) / 2

            # Backprop and step
            loss_d.backward()
            opts.step(['discriminator'], False)

            # Saving
            batches_done = epoch * len(dataloader) + i
            if batches_done % cfg.LOG.SAVE_ITER == 0:
                print("Saving samples and weights")
                sample_image(cgn, sample_path, batches_done, device, n_row=3)
                torch.save(cgn.state_dict(), f"{weights_path}/ckp_{batches_done:d}.pth")

            # Logging
            if cfg.LOG.LOSSES:
                msg = f"[Batch {i}/{len(dataloader)}]"
                msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_d.items()])
                msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
                pbar.set_description(msg)

def main(cfg):
    # model init
    cgn = CGN(n_classes=cfg.MODEL.N_CLASSES, latent_sz=cfg.MODEL.LATENT_SZ,
              ngf=cfg.MODEL.NGF, init_type=cfg.MODEL.INIT_TYPE,
              init_gain=cfg.MODEL.INIT_GAIN)
    Discriminator = DiscLin if cfg.MODEL.DISC == 'linear' else DiscConv
    discriminator = Discriminator(n_classes=cfg.MODEL.N_CLASSES, ndf=cfg.MODEL.NDF)

    # get data
    dataloader, _ = get_dataloaders(cfg.TRAIN.DATASET, cfg.TRAIN.BATCH_SIZE,
                                    cfg.TRAIN.WORKERS)

    # Loss functions
    L_adv = torch.nn.MSELoss()
    L_binary = BinaryLoss(cfg.LAMBDAS.MASK)
    L_perc = PerceptualLoss(style_wgts=cfg.LAMBDAS.PERC)
    losses = (L_perc, L_adv, L_binary)

    # Optimizers
    opts = Optimizers()
    opts.set('generator', cgn, lr=cfg.LR.LR, betas=cfg.LR.BETAS)
    opts.set('discriminator', discriminator, lr=cfg.LR.LR, betas=cfg.LR.BETAS)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = cgn.to(device)
    discriminator = discriminator.to(device)
    losses = (l.to(device) for l in losses)

    fit(cfg, cgn, discriminator, dataloader, opts, losses, device)

def merge_args_and_cfg(args, cfg):
    cfg.MODEL_NAME = args.model_name
    cfg.LOG.SAVE_ITER = args.save_iter
    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='',
                        help="path to a cfg file")
    parser.add_argument('--model_name', default='tmp',
                        help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument("--save_iter", type=int, default=1000,
                        help="interval between image sampling")
    parser.add_argument("--epochs", type=int, default=15,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="size of the batches")
    args = parser.parse_args()

    # get cfg
    cfg = load_cfg(args.cfg) if args.cfg else get_cfg_defaults()
    # add additional arguments in the argparser and in the function below
    cfg = merge_args_and_cfg(args, cfg)

    print(cfg)
    main(cfg)
