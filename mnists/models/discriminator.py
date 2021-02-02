import numpy as np
import torch
from torch import nn

from utils import init_net

class DiscLin(nn.Module):
    def __init__(self, n_classes, ndf, img_shape=[3, 32, 32]):
        super(DiscLin, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, ndf),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, ndf),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity.squeeze()

class DiscConv(nn.Module):
    def __init__(self, n_classes, ndf):
        super(DiscConv, self).__init__()
        cin = 4  # RGB + Embedding
        self.label_embedding = nn.Embedding(n_classes, 1)

        def block(cin, cout, ks, st):
            return[
                nn.Conv2d(cin, cout, ks, stride=st, padding=0, bias=False),
                nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.2, True),
            ]

        self.model = nn.Sequential(
            *block(cin, ndf, 3, 1),
            *block(ndf, ndf*2, 3, 1),
            *block(ndf*2, ndf*4, 4, 2),
            *block(ndf*4, ndf*4, 4, 2),
            nn.AvgPool2d(3),
            nn.Conv2d(ndf*4, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )
        init_net(self)

    def forward(self, ims, labels):
        # build embedding channel
        embedding = self.label_embedding(labels)
        embedding = embedding.reshape(-1, 1, 1, 1)
        embedding = embedding.repeat(1, 1, *ims.shape[-2:])

        out = self.model(torch.cat([ims, embedding], 1))
        return out.squeeze()
