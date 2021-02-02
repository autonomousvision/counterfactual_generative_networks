import torch
import torch.nn as nn

class Patch2Image(nn.Module):
    ''' take in patch and copy n_up times to form the full image'''
    def __init__(self, patch_sz,  n_up):
        super(Patch2Image, self).__init__()
        self.patch_sz = patch_sz
        self.n_up = n_up

    def forward(self, x):
        assert x.shape[-1]==self.patch_sz, f"inp.patch_sz ({x.shape[-1]}): =/= self.patch_sz ({self.patch_sz})"
        x = torch.cat([x]*self.n_up, -1)
        x = torch.cat([x]*self.n_up, -2)
        return x

def choose_rand_patches(x, patch_sz, dim):
    assert dim == 2 or dim == 3
    batch_sz = x.shape[0]

    # get all possible patches
    patches = x.unfold(dim, patch_sz, 1)
    n_patches = patches.shape[2]

    # for each image, choose a random patch
    idx = torch.randint(0, n_patches, (batch_sz,))

    if dim == 2:
        patches = patches[torch.arange(batch_sz), :, idx, :]
    elif dim == 3:
        patches = patches[torch.arange(batch_sz), :, :, idx]
    return patches

class RandomCrop(nn.Module):
    def __init__(self, crop_sz):
        super(RandomCrop, self).__init__()
        self.crop_sz = crop_sz

    def forward(self, x):
        img_sz = x.shape[-1]
        assert img_sz >= self.crop_sz, f"img_sz {img_sz} is too small for crop_sz {self.crop_sz}"
        x = choose_rand_patches(x, self.crop_sz, 2)
        x = choose_rand_patches(x, self.crop_sz, 2)
        return x
