import os
from os.path import join
from math import ceil
from glob import glob

from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

class Flatten(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class InvariantEnsemble(nn.Module):

    def __init__(self, arch, pretrained, cut=1):
        super(InvariantEnsemble, self).__init__()

        def get_backbone(cut):
            m = model_arch(pretrained)
            return nn.Sequential(*list(m.children())[:-cut])

        def get_head(cut):
            m = model_arch(False)
            ch = list(m.children())
            back = ch[-cut:-1]
            head = ch[-1]
            return nn.Sequential(*back, Flatten(), head)

        model_arch = models.__dict__[arch]

        # cut the classifier head and reinit
        # you can also set this higher to get a deeper classifier
        # We find that cut=1 yields the best results
        self.backbone = get_backbone(cut)
        self.m_shape = get_head(cut)
        self.m_texture = get_head(cut)
        self.m_bg = get_head(cut)

    def forward(self, x):
        feats = self.backbone(x)

        shape_preds = self.m_shape(feats)
        texture_preds = self.m_texture(feats)
        bg_preds = self.m_bg(feats)

        # average the logits
        avg_preds = (shape_preds + texture_preds + bg_preds) / 3

        # pred without the background signal
        shape_texture_preds = (shape_preds + texture_preds) / 2

        return {
            'shape_preds': shape_preds,
            'texture_preds': texture_preds,
            'bg_preds': bg_preds,
            'avg_preds': avg_preds,
            'shape_texture_preds': shape_texture_preds,
        }
