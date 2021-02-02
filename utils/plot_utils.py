import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

def plot_tensor(t, backend='matplot', sz=80):
    if backend=='pil':
        return to_pil_image(t).resize((sz, sz))
    elif backend=='matplot':
        grid = make_grid(t.cpu().unsqueeze(0), nrow=1, padding=2, normalize=True)
        plt.imshow(np.transpose(grid,(1, 2, 0)))
        plt.axis('off')
    else:
        raise TypeError(f"Unknown backend {backend}")

def plot_batch(ims, figsize=(5, 10), per_row=4, normalize=True):
    grid = make_grid(ims, nrow=ims.shape[0]//per_row, padding=2, normalize=normalize)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(np.transpose(grid,(1, 2, 0)))