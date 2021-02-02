import torch
from torch import nn

from utils import get_norm_layer, init_net, choose_rand_patches, Patch2Image, RandomCrop

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def UpsampleBlock(cin, cout, scale_factor=2):
    return [
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    ]

def lin_block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

def shape_layers(cin, cout, ngf, init_sz):
    return [
        nn.Linear(cin, ngf*2 * init_sz ** 2),
        Reshape(*(-1, ngf*2, init_sz, init_sz)),
        get_norm_layer()(ngf*2),
        *UpsampleBlock(ngf*2, ngf),
        get_norm_layer()(ngf),
        *UpsampleBlock(ngf, cout),
        get_norm_layer()(cout),
    ]

def texture_layers(cin, cout, ngf, init_sz):
    return [
        nn.Linear(cin, ngf*2 * init_sz ** 2),
        Reshape(*(-1, ngf*2, init_sz, init_sz)),
        *UpsampleBlock(ngf*2, ngf*2),
        *UpsampleBlock(ngf*2, ngf),
        nn.Conv2d(ngf, cout, 3, stride=1, padding=1),
        nn.BatchNorm2d(cout),
    ]

class CGN(nn.Module):

    def __init__(self, n_classes=10, latent_sz=32, ngf=32,
                 init_type='orthogonal', init_gain=0.1, img_sz=32):
        super(CGN, self).__init__()

        # params
        self.batch_size = 1  # default: sample a single image
        self.n_classes = n_classes
        self.latent_sz = latent_sz
        self.label_emb = nn.Embedding(n_classes, n_classes)
        init_sz = img_sz // 4
        inp_dim = self.latent_sz + self.n_classes

        # models
        self.f_shape = nn.Sequential(*shape_layers(inp_dim, 1, ngf, init_sz))
        self.f_text1 = nn.Sequential(*texture_layers(inp_dim, 3, ngf, init_sz), nn.Tanh())
        self.f_text2 = nn.Sequential(*texture_layers(inp_dim, 3, ngf, init_sz), nn.Tanh())
        self.shuffler = nn.Sequential(Patch2Image(img_sz, 2), RandomCrop(img_sz))

        init_net(self, init_type=init_type, init_gain=init_gain)

    def get_inp(self, ys):
        u_vec = torch.normal(0, 1, (len(ys), self.latent_sz)).to(ys.device)
        y_vec = self.label_emb(ys)
        return torch.cat([u_vec, y_vec], -1)

    def forward(self, ys=None, counterfactual=False):

        if counterfactual:
            # randomize labels for fore- and background
            inp0 = self.get_inp(ys)
            inp1 = self.get_inp(ys[torch.randperm(len(ys))])
            inp2 = self.get_inp(ys[torch.randperm(len(ys))])
        else:
            inp = self.get_inp(ys)
            inp0, inp1, inp2 = inp, inp, inp

        # Masker
        mask = self.f_shape(inp0)
        mask = torch.clamp(mask, 0.0001, 0.9999).repeat(1, 3, 1, 1)

        # foreground & background
        foreground = self.shuffler(self.f_text1(inp1))
        background = self.shuffler(self.f_text2(inp2))

        return mask, foreground, background
