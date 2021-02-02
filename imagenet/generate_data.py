'''
Generate a dataset with the CGN.
The labels are stored in a csv
'''

import warnings
import pathlib
from os.path import join
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import torch
from torch import nn
import torchvision
from torchvision.utils import make_grid
import repackage
repackage.up()

from imagenet.models import CGN

def save_image(im, path):
    torchvision.utils.save_image(im.detach().cpu(), path, normalize=True)

def interp(x0, x1, num_midpoints):
    '''
    Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
    '''
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device=x0.device).to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))

def interp_sheet(cgn, mode, ys, y_interp, num_midpoints, save_path,
                 save_single=False, save_noise=True):
    dev = cgn.get_device()

    if y_interp == -1:
        y_interp = np.random.randint(0, 1000)

    # Prepare zs
    dim_u = cgn.dim_u
    if save_noise:
        print('Sampling new noise and saving.')
        us = cgn.get_noise_vec(sz=1)
        torch.save(us, 'imagenet/data/u_fixed.pth')
    else:
        print('Loading noise vector from disk.')
        us = torch.load('imagenet/data/u_fixed.pth')

    us_stack = us.repeat(1, num_midpoints + 2, 1).view(-1, dim_u).to(dev)

    def stack_vec(y_start):
        y_start_vec = cgn.get_class_vec(y=y_start, sz=1)
        return y_start_vec.repeat(num_midpoints + 2, 1)

    def interp_vec(y_start, y_end):
        y_start_vec = cgn.get_class_vec(y=y_start, sz=1)
        y_end_vec = cgn.get_class_vec(y=y_end, sz=1)
        return interp(y_start_vec.view(1, 1, -1), y_end_vec.view(1, 1, -1),
                      num_midpoints).view((num_midpoints + 2), -1)

    # get class vectors
    if mode == 'shape':
        y_vec_0 = interp_vec(ys[0], y_interp)
        y_vec_1 = stack_vec(ys[1])
        y_vec_2 = stack_vec(ys[2])

    elif mode == 'text':
        y_vec_0 = stack_vec(ys[0])
        y_vec_1 = interp_vec(ys[1], y_interp)
        y_vec_2 = stack_vec(ys[2])

    elif mode == 'bg':
        y_vec_0 = stack_vec(ys[0])
        y_vec_1 = stack_vec(ys[1])
        y_vec_2 = interp_vec(ys[2], y_interp)

    elif mode == 'all':
        y_vec_0 = interp_vec(ys[0], y_interp)
        y_vec_1 = interp_vec(ys[1], y_interp)
        y_vec_2 = interp_vec(ys[2], y_interp)

    y_vec_0, y_vec_1, y_vec_2 = y_vec_0.to(dev), y_vec_1.to(dev), y_vec_2.to(dev)
    premask, mask, foreground, background = [], [], [], []

    # loop over each datapoint, otherwise we need to much GPU mem when using many midpoints
    with torch.no_grad():
        for i in trange(len(y_vec_0)):
            # partially copying the forward pass of the cgn class
            pm = cgn.f_shape(us_stack[i][None], y_vec_0[i][None], cgn.truncation)
            m = cgn.u2net(pm)
            m = torch.clamp(m, 0.0001, 0.9999).repeat(1, 3, 1, 1)
            fg = cgn.f_text(us_stack[i][None], y_vec_1[i][None], cgn.truncation)
            bg = cgn.f_bg(us_stack[i][None], y_vec_2[i][None], cgn.truncation)

            # build lists
            premask.append(pm)
            mask.append(m)
            foreground.append(fg)
            background.append(bg)

        # compose
        premask, mask = torch.cat(premask), torch.cat(mask)
        foreground, background = torch.cat(foreground), torch.cat(background)
        x_gen = mask * foreground + (1 - mask) * background

    # get the result of the IM we interpolated over, e.g., only shape
    if mode == 'shape': out = mask
    elif mode == 'text': out = foreground
    elif mode == 'bg': out = background
    elif mode == 'all': out = x_gen

    # save composite image
    x_gen_file = f'{save_path}_x_gen_interp.jpg'
    out_file = f'{save_path}_{mode}_interp.jpg'
    if not save_single:
        torchvision.utils.save_image(x_gen, x_gen_file, nrow=num_midpoints + 2, normalize=True)
        if mode != 'all':
            torchvision.utils.save_image(out, out_file, nrow=num_midpoints + 2, normalize=True)
    else:
        for i in range(len(x_gen)):
            idx = str(i).zfill(5)
            save_image(x_gen[i], x_gen_file.replace('.jpg', idx + '.jpg'))

# Lists of best or most interesting shape/texture/background classes
# (Yes, I know all imagenet classes very well by now)
MASKS = [9, 18, 22, 35, 56, 63, 96, 97, 119, 207, 225, 260, 275, 323, 330, 350, 370, 403, 411,
         414, 427, 438, 439, 441, 460, 484, 493, 518, 532, 540, 550, 559, 561, 570, 604, 647,
         688, 713, 724, 749, 751, 756, 759, 779, 780, 802, 814, 833, 841, 849, 850, 859, 869,
         872, 873, 874, 876, 880, 881, 883, 894, 897, 898, 900, 907, 930, 933, 945, 947, 949,
         950, 953, 963, 966, 967, 980]
FOREGROUND = [12, 15, 18, 25, 54, 66, 72, 130, 145, 207, 251, 267, 271, 275, 293, 323, 385,
              388, 407, 409, 427, 438, 439, 441, 454, 461, 468, 482, 483, 486, 490, 492, 509,
              530, 555, 607, 608, 629, 649, 652, 681, 688, 719, 720, 728, 737, 741, 751, 756,
              779, 800, 810, 850, 852, 854, 869, 881, 907, 911, 930, 936, 937, 938, 941, 949,
              950, 951, 954, 957, 959, 963, 966, 985, 987, 992]
BACKGROUNDS = [7, 9, 20, 30, 35, 46, 50, 65, 72, 93, 96, 97, 119, 133, 147, 337, 350, 353, 354,
               383, 429, 460, 693, 801, 888, 947, 949, 952, 953, 955, 958, 970, 972, 973, 974,
               977, 979, 998]

def sample_classes(mode, classes=None):
    if mode == 'random':
        return np.random.randint(0, 1000, 3).tolist()

    elif mode == 'best_classes':
        return [np.random.choice(MASKS),
                np.random.choice(FOREGROUND),
                np.random.choice(BACKGROUNDS)]

    elif mode == 'fixed_classes':
        return [int(c) for c in classes]

    else:
        assert ValueError("Unknown sample mode {mode}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    cgn = CGN(
        batch_sz=args.batch_sz,
        truncation=args.truncation,
        pretrained=False,
    )
    cgn = cgn.eval().to(device)

    weights = torch.load(args.weights_path, map_location='cpu')
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    cgn.load_state_dict(weights)
    cgn.eval().to(device)

    # path setup
    time_str = datetime.now().strftime("%Y_%m_%d_%H_")
    trunc_str = f"{args.run_name}_trunc_{args.truncation}"
    data_path = join('imagenet', 'data', time_str + trunc_str)
    ims_path = join(data_path, 'ims')
    pathlib.Path(ims_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving data to {data_path}")

    # set up label csv
    df = pd.DataFrame(columns=['im_name', 'shape_cls', 'texture_cls', 'bg_cls'])
    csv_path = join(data_path, 'labels.csv')
    df.to_csv(csv_path)

    # generate data
    with torch.no_grad():
        for i in trange(args.n_data):
            # sample class vector and set up the save path
            ys = sample_classes(args.mode, args.classes)
            im_name = f'{args.run_name}_{i:07}'

            if args.interp:
                # interpolation between the first and second class in the class vector
                interp_sheet(cgn=cgn, mode=args.interp, ys=ys, y_interp=args.interp_cls,
                             num_midpoints=args.midpoints, save_path=join(ims_path, im_name),
                             save_single=args.save_single, save_noise=args.save_noise)
            else:
                x_gt, mask, premask, foreground, background, bg_mask = cgn(ys=ys)
                x_gen = mask * foreground + (1 - mask) * background

                # save image
                # to save other outputs, simply add a line in the same format, e.g.:
                # save_image(premask, join(ims_path, im_name + '_premask.jpg'))
                save_image(x_gen, join(ims_path, im_name + '_x_gen.jpg'))

            # save labels
            df = pd.DataFrame(columns=[im_name] + ys)
            df.to_csv(csv_path, mode='a')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['random', 'best_classes', 'fixed_classes'],
                        help='Choose between random sampling, sampling from the best ' +
                        'classes or the classes passed to args.classes')
    parser.add_argument('--n_data', type=int, required=True,
                        help='How many datapoints to sample')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name the dataset')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Which weights to load for the CGN')
    parser.add_argument('--batch_sz', type=int, default=1,
                        help='For generating a dataset, keep this at 1')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for the sampling the noise')
    parser.add_argument('--classes', nargs='+', default=[0, 0, 0],
                        help='Classes to sample from, use in conjunction with ' +
                        'args.mode=fixed_classes. Order: [Shape, Foreground, Background]')
    parser.add_argument('--interp', type=str, default='',
                        choices=['', 'all', 'shape', 'text', 'bg'],
                        help='Save interpolation sheet instead of single ims.')
    parser.add_argument('--interp_cls', type=int, default=-1,
                        help='Classes to which we interpolate. val=-1 samples a random class.')
    parser.add_argument('--midpoints', type=int, default=6,
                        help='How many midpoints for the interpolation')
    parser.add_argument('--save_noise', default=False, action='store_true',
                        help='Sample new noise and save to disk for interpolation')
    parser.add_argument('--save_single', default=False, action='store_true',
                        help='Sample single images instead of sheets')

    args = parser.parse_args()
    print(args)
    if args.mode != 'fixed_classes' and [0, 0, 0] != args.classes:
        warnings.warn(f"You supply classes, but they won't be used for mode = {args.mode}")
    main(args)
