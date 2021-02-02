import os

from scipy.stats import truncnorm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from imagenet.models import BigGAN, U2NET
from utils import toggle_grad

class CGN(nn.Module):
    def __init__(self, batch_sz, truncation=0.5, pretrained=True):
        super(CGN, self).__init__()

        self.dim_u = 128
        self.truncation = truncation
        self.batch_sz = batch_sz
        self.cl = None

        # pretrained weights
        biggan_weights = 'imagenet/weights/biggan256.pth' if pretrained else None
        u2net_weights = 'imagenet/weights/u2net.pth' if pretrained else None

        # The unconstrained cGAN
        self.biggan_GT = BigGAN.initialize(biggan_weights).eval()
        toggle_grad(self.biggan_GT, False)

        # The independent mechanisms
        self.f_shape = BigGAN.initialize(biggan_weights)
        self.f_text = BigGAN.initialize(biggan_weights)
        self.f_bg = BigGAN.initialize(biggan_weights)

        # U2net for processing pre-masks to masks and for the background loss L_bg
        self.u2net = U2NET.initialize(u2net_weights).eval()
        toggle_grad(self.u2net, False)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        # cGAN and U2net should remain in eval mode
        self.biggan_GT.eval()
        self.u2net.eval()
        return self

    def get_device(self):
        return list(self.parameters())[0].device

    @staticmethod
    def truncated_noise_sample(batch_size=1, dim_z=128, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state)
        values = values.astype(np.float32)
        return truncation * values

    def get_noise_vec(self, sz=None):
        if sz is None: sz = self.batch_sz
        u = self.truncated_noise_sample(batch_size=sz, truncation=self.truncation)
        return torch.from_numpy(u)

    def get_class_vec(self, y, sz=None):
        if sz is None: sz = self.batch_sz
        y_vec = y * torch.ones(sz).to(torch.int64)
        y_vec = F.one_hot(y_vec, 1000).to(torch.float32)
        return y_vec

    def get_inp(self, ys=None):
        if ys is None:
            ys = 3 * [np.random.randint(0, 1000)]

        dev = self.get_device()
        u_vec = self.get_noise_vec()
        inp0 = (u_vec.to(dev), self.get_class_vec(y=ys[0]).to(dev), self.truncation)
        inp1 = (u_vec.to(dev), self.get_class_vec(y=ys[1]).to(dev), self.truncation)
        inp2 = (u_vec.to(dev), self.get_class_vec(y=ys[2]).to(dev), self.truncation)
        return inp0, inp1, inp2

    def forward(self, inp=None, ys=None):
        """
        three possible options for a forward pass:
            1. cgn(): randomly choose classes, it is the same class
               for all IMs (the standard mode for training
            2. cgn(inp=(u, y, trunc)): sample input before pass, useful
               for fixed noise samples
            3. cgn(ys=[10, 5, 32]): list with 3 classes, a class for
               every IM (m, fg, bg)
        """
        if inp is None:
            if ys is not None:
                assert len(ys) == 3, 'Provide 3 classes'
            inp0, inp1, inp2 = self.get_inp(ys)
        else:
            inp0, inp1, inp2 = inp, inp, inp

        # cGAN
        x_gt = self.biggan_GT(*inp0).detach()

        #  Masker
        premask = self.f_shape(*inp0)
        mask = self.u2net(premask)
        mask = torch.clamp(mask, 0.0001, 0.9999)

        # Texture
        foreground = self.f_text(*inp1)

        # Background
        background = self.f_bg(*inp2)
        background_mask = self.u2net(background)

        return x_gt, mask, premask.detach(), foreground, background, background_mask
