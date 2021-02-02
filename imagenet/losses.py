import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid

from utils import requires_grad, children, hook_outputs

# helper functions

def get_all_patches(ims, patch_sz=[10, 10], pad=True):
    '''
    given a batch of images, get the surrounding patch (of patch_sz=(height,width)) for each pixel
    '''
    assert isinstance(patch_sz, list) and len(patch_sz) == 2, "Wrong format for patch_sz"

    # Pad the images - we want the patch surround each pixel
    patch_sz = np.array(patch_sz)
    patch_sz += (patch_sz+1) % 2  # round up to odd number

    # padding if we want to get the surrounding patches for *all* pixels
    if pad:
        pad = tuple((patch_sz//2).repeat(2))
        ims = nn.ReflectionPad2d(pad)(ims)

    # unfold the last 2 dimensions to get all patches
    patches = ims.unfold(2, patch_sz[0], 1).unfold(3, patch_sz[1], 1)

    # reshape to no_pixel x c x patch_sz x patch_sz
    batch_sz, c, w, h = patches.shape[:4]
    patch_batch = patches.reshape(batch_sz, c, w*h, patch_sz[0], patch_sz[1])
    patch_batch = patch_batch.permute(0, 2, 1, 3, 4)
    patch_batch = patch_batch.reshape(batch_sz*w*h, c, patch_sz[0], patch_sz[1])

    if pad: assert patch_batch.shape[0] == batch_sz * w * h  # one patch per pixel per image

    return patch_batch

def get_sampled_patches(prob_maps, paint, patch_sz=[30, 30], sample_sz=500, n_up=None):
    paint_shape = paint.shape[-2:]
    prob_maps = F.interpolate(prob_maps, (128, 128), mode='bicubic', align_corners=False)
    paint = F.interpolate(paint, (128, 128), mode='bicubic', align_corners=False)

    mode_patches = []
    if n_up is None:
        n_up = paint.shape[-1]//patch_sz[0]

    for p, prob in zip(paint, prob_maps):
        prob_patches = get_all_patches(prob.unsqueeze(0), patch_sz, pad=False)
        prob_patches_mean = prob_patches.mean((1, 2, 3))
        max_ind = torch.argsort(prob_patches_mean)[-sample_sz:]  # get 400 topvalues
        max_ind = max_ind[torch.randint(len(max_ind), (n_up**2,))].squeeze()  # sample one
        p_patches = get_all_patches(p[None], patch_sz, pad=False)
        patches = p_patches[max_ind]
        patches = make_grid(patches, nrow=n_up, padding=0)
        patches = F.interpolate(patches[None], paint_shape, mode='bicubic', align_corners=False)
        mode_patches.append(patches)

    return torch.cat(mode_patches)

# adversarial losses

class BigGAN_Loss(nn.Module):
    def __init__(self, loss_weight, mode='dcgan'):
        super(BigGAN_Loss, self).__init__()
        self.loss_weight = loss_weight

        if mode == 'dcgan':
            self.loss_d = self.loss_dcgan_dis
            self.loss_g = self.loss_dcgan_gen

        elif mode == 'hinge':
            self.loss_d = self.loss_hinge_dis
            self.loss_g = self.loss_hinge_gen

    @staticmethod
    def loss_hinge_dis(dis_fake, dis_real):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real + loss_fake

    @staticmethod
    def loss_hinge_gen(dis_fake):
        loss = -torch.mean(dis_fake)
        return loss

    @staticmethod
    def loss_dcgan_dis(dis_fake, dis_real):
        loss_real = torch.mean(F.softplus(-dis_real))
        loss_fake = torch.mean(F.softplus(dis_fake))
        return loss_real + loss_fake

    @staticmethod
    def loss_dcgan_gen(dis_fake):
        loss = torch.mean(F.softplus(-dis_fake))
        return loss

    def __call__(self, D, x, y, x_gen, y_gen=None, train_G=False):
        if not train_G:
            assert y_gen is not None, "supply pseudo labels for training D"

        if train_G:
            fake = D(x_gen, y)
            loss = self.loss_weight * self.loss_g(fake)  # apply loss scaling only for G
        else:
            out = D(torch.cat([x_gen, x]), torch.cat([y_gen, y]))
            fake, real = torch.split(out, [x_gen.shape[0], x.shape[0]])
            loss = self.loss_d(fake, real)

        return loss

# reconstruction losses

class ReconstructionLoss(nn.Module):
    def __init__(self, mode, loss_weight):
        super(ReconstructionLoss, self).__init__()
        self.mode = mode
        self.loss_weight = loss_weight

        if mode == 'l2':
            self.loss = nn.MSELoss()
        elif mode == 'l1':
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError(f'Mode {mode} not implemented')

    def __call__(self, x_rec, x):
        return self.loss_weight * self.loss(x_rec, x)

# perceptual losses

class ParallelPercLoss(torch.nn.DataParallel):
    def __init__(self, args):
        super(ParallelPercLoss, self).__init__(args)
        self.output_device='cuda:1'
        self.replicas = self.replicate(self.module, self.device_ids)

    def __len__(self):
        return len(self.replicas)

    def __call__(self, x_rec, x):
        x_rec_scattered, _ = self.scatter((x_rec,), {}, self.device_ids)
        x_scatter, _ = self.scatter((x,), {}, self.device_ids)
        outputs = [self.replicas[i](x_rec_scattered[i][0], x_scatter[i][0]) for i in range(len(self))]
        return self.gather(outputs, self.output_device)

class PerceptualLossModel(nn.Module):

    def __init__(self, m_feat, layer_ids, cont_wgts, style_wgts):
        super().__init__()
        self.base_loss = F.l1_loss

        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.cont_wgts = cont_wgts
        self.style_wgts = style_wgts

        self.metric_names = [f'feat_{i}' for i in range(len(layer_ids))]
        self.metric_names += [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(f_in, f_out)*w
                            for f_in, f_out, w in zip(in_feat, out_feat, self.cont_wgts)]
        self.feat_losses += [self.base_loss(self.gram_matrix(f_in), self.gram_matrix(f_out))*w**2*5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.style_wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

    @staticmethod
    def gram_matrix(x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return (x @ x.transpose(1, 2))/(c*h*w)

class PerceptualLoss(nn.Module):

    def __init__(self, cont_wgts=[0, 0, 0, 0], style_wgts=[0, 0, 0, 0]):
        super().__init__()
        pretrained_model = torchvision.models.vgg16(True).features.eval()
        requires_grad(pretrained_model, False)
        blocks = [i-1 for i, o in enumerate(children(pretrained_model)) if isinstance(o, nn.MaxPool2d)]
        self.model = PerceptualLossModel(pretrained_model, blocks[:4], cont_wgts, style_wgts)

    def __call__(self, x_rec, x):
        return self.model(x_rec, x)

class PercLossText(nn.Module):
    def __init__(self, style_wgts, patch_sz=[15, 15], im_sz=256, sample_sz=100, n_up=6):
        super(PercLossText, self).__init__()
        self.loss = PerceptualLoss(style_wgts=style_wgts)
        self.patch_sz = patch_sz
        self.sample_sz = sample_sz
        self.n_up = n_up

    def forward(self, ims, mask, paint):
        ims, mask = ims.detach(), mask.detach()
        paint_temp = get_sampled_patches(mask, ims, self.patch_sz, self.sample_sz, n_up=self.n_up)
        return self.loss(paint, paint_temp.detach())

# mask losses

class BinaryLoss(nn.Module):

    def __init__(self, loss_weight):
        super(BinaryLoss, self).__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def binary_entropy(p):
        return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)

    def __call__(self, mask):
        return self.loss_weight * self.binary_entropy(mask).mean()

class MaskLoss(nn.Module):

    def __init__(self, loss_weight, interval=[0.1, 0.9]):
        super(MaskLoss, self).__init__()
        self.loss_weight = loss_weight
        self.interval = interval

    def __call__(self, mask):
        mask_mean = torch.mean(mask, [1, 2, 3])
        zero = torch.zeros(1).to(mask.device)
        loss_min = torch.max(zero, self.interval[0] - mask_mean)
        loss_max = torch.max(zero, mask_mean - self.interval[1])
        return self.loss_weight * (loss_min + loss_max)

# background loss

class BackgroundLoss(nn.Module):

    def __init__(self, loss_weight):
        super(BackgroundLoss, self).__init__()
        self.loss_weight = loss_weight

    def __call__(self, mask):
        loss = self.loss_weight * mask.reshape(mask.shape[0], -1).mean(1)
        loss = torch.max(torch.zeros_like(loss), loss)
        return loss
