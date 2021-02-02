import re
import math
import os
from os.path import join
from math import ceil
from glob import glob
from pathlib import PurePosixPath

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import Sampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# helper functions

class DistributedSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # split to the nearest length that is evenly divisible. This is to ensure
            # that each rank gets the same amount of data when iterating this
            # dataloader.
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

def transform_labels(x):
    return torch.tensor(x).to(torch.int64)

# datasets

class ImagenetVanilla(Dataset) :

    def __init__(self, train=True):
        super(ImagenetVanilla, self).__init__()
        root = join('.', 'imagenet', 'data')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Transforms
        if train:
            ims_path = join(root, 'imagenet', 'train')
            t_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            ims_path = join(root, 'imagenet', 'val')
            t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

        t_list += [transforms.ToTensor(), normalize]
        self.T_ims = transforms.Compose(t_list)

        self.im_paths, self.labels = self.get_data(ims_path)

    def set_len(self, n):
        assert n < len(self), "Ratio is too large, not enough CF data available"
        self.im_paths = self.im_paths[:n]
        self.labels = self.labels[:n]

    @staticmethod
    def get_data(p):
        ims, labels = [], []
        subdirs = sorted(glob(p + '/*'))
        for i, sub in enumerate(subdirs):
            im = sorted(glob(sub + '/*'))
            l = np.ones(len(im))*i
            ims.append(im), labels.append(l)
        return np.concatenate(ims), np.concatenate(labels)

    def __getitem__(self, idx):
        ims = Image.open(self.im_paths[idx]).convert('RGB')
        labels = self.labels[idx]
        return {
            'ims': self.T_ims(ims),
            'labels': transform_labels(labels),
        }

    def __len__(self):
        return len(self.im_paths)

class ImagenetCounterfactual(Dataset):
    '''
    args:
        - ims_path: the relative path to the saved dataset, e.g.,
          "imagenet/data/2021_01_25_16_counterfactuals_trunc_0.5"
        - train: choose subfolder with "train" or "val" in name
        - n_data: how many images to use from the whole dataset.
        - mode: default is 'x_gen'. If you generate, e.g., only texture
          change mode to 'textures'. Then, images should have the name format
          "RUN_NAME_0000000_textures.jpg"
    '''

    def __init__(self, ims_path, train=True, n_data=None, mode='silhouette'):
        super(ImagenetCounterfactual, self).__init__()
        print(f"Loading counterfactual data from {ims_path}")
        self.full_df = self.get_data(ims_path, train, mode)

        # get the actual dataframe that we use for sampling the dataset
        if n_data is None: n_data = len(self.full_df)
        self.n_data = n_data
        if n_data > len(self.full_df):
            mult = ceil(n_data/len(self.full_df))
            self.full_df = pd.concat(mult * [self.full_df])
        # setting df for the first time
        self.resample()
        print(f"=> Current dataset sz: {len(self.df)}. Full dataset sz: {len(self.full_df)}")

        # Transforms
        if train:
            t_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
        else:
            t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

        t_list += [transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]
        self.T_ims = transforms.Compose(t_list)

    def resample(self):
        ''' resample the current data set from the full dataset'''
        self.df = self.full_df.sample(self.n_data)

    @staticmethod
    def get_data(p, train , mode):
        subdirs = glob(p + '/train*') if train else glob(p + '/val*')

        dfs = []
        for sub in subdirs:
            df = pd.read_csv(join(sub, 'labels.csv'), index_col=0)
            df['abs_path'] = sub + '/ims/' + df['im_name'] + f"_{mode}.jpg"
            dfs.append(df)

        return pd.concat(dfs)

    def __getitem__(self, idx):
        # get image
        im_path = self.df.abs_path.iloc[idx]
        ims = Image.open(im_path).convert('RGB')

        # get labels
        shape_labels = self.df.shape_cls.iloc[idx]
        texture_labels = self.df.texture_cls.iloc[idx]
        bg_labels = self.df.bg_cls.iloc[idx]

        return {
            'ims': self.T_ims(ims),
            'shape_labels': transform_labels(shape_labels),
            'texture_labels': transform_labels(texture_labels),
            'bg_labels': transform_labels(bg_labels),
        }

    def __len__(self):
        return len(self.df)

class CueConflict(Dataset):
    def __init__(self, t_list=[transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]):
        super(CueConflict, self).__init__()
        path = 'imagenet/data/cue_conflict/'
        self.df = self.make_df(path)
        self.T = transforms.Compose(t_list)

    def __getitem__(self, idx):
        # get image
        im_path = self.df.abs_path.iloc[idx]
        ims = Image.open(im_path).convert('RGB')

        # get labels
        shape_labels = self.df.shape_cls.iloc[idx]
        texture_labels = self.df.texture_cls.iloc[idx]

        return {
            'ims': self.T(ims),
            'shape_labels': shape_labels,
            'texture_labels': texture_labels,
        }

    @staticmethod
    def make_df(path):
        # get the image paths
        subdirs = glob(path + '*')
        ims = []
        for sub in subdirs: ims += glob(sub + '/*')

        # get the labels
        def first_label(l): return l[:re.search("\d+-",l).start()]
        def last_label(l): return l[re.search("-",l).start()+1:-5]
        shapes = [first_label(PurePosixPath(s).parts[-1]) for s in ims]
        textures = [last_label(PurePosixPath(s).parts[-1]) for s in ims]

        # labels to indicies
        from utils import get_human_object_recognition_categories
        categories = get_human_object_recognition_categories()
        name2idx = {c: i for c, i in zip(categories, range(len(categories)))}

        # compile the datafrem and kickout duplicates
        df = pd.DataFrame([ims, shapes, textures]).T
        df.columns = ['abs_path', 'shape_cls', 'texture_cls']
        df = df[df.shape_cls != df.texture_cls]
        return df

    def __len__(self):
        return len(self.df)

class Imagenet9(object):

    def __init__(self, data_path):
        self.ds_name = 'ImageNet9'
        self.data_path = data_path
        self.num_classes = 9

        mean = torch.tensor([0.4717, 0.4499, 0.3837])
        std = torch.tensor([0.2600, 0.2516, 0.2575])
        self.T = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])

    def make_loader(self, distributed, workers, batch_size):
        print(f"==> Preparing dataset {self.ds_name}..")
        test_path = os.path.join(self.data_path, 'val')
        if not os.path.exists(test_path):
            raise ValueError("Test data must be stored in {0}".format(test_path))

        test_set = ImageFolder(root=test_path, transform=self.T)
        sampler = DistributedSampler(test_set, drop_last=True, shuffle=False) if distributed else None
        test_loader = DataLoader(test_set, batch_size=batch_size, sampler=sampler,
                                 shuffle=False, num_workers=workers, pin_memory=True)
        return test_loader

# dataloaders

def get_imagenet_dls(distributed, batch_size, workers):
    # dataset
    train_dataset = ImagenetVanilla(train=True)
    val_dataset = ImagenetVanilla(train=False)

    # sampler
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, drop_last=True, shuffle=False) if distributed else None

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=(train_sampler is None), num_workers=workers,
                              pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader, train_sampler

def get_cf_imagenet_dls(path, cf_ratio, len_dl_train, distributed, batch_size, workers):
    # determine how many images to use, based on given ratio
    cf_batch_sz = int(cf_ratio * batch_size)
    n_data = cf_batch_sz * len_dl_train

    # dataset
    cf_train_dataset = ImagenetCounterfactual(path, train=True, n_data=n_data)
    cf_val_dataset = ImagenetCounterfactual(path, train=False)

    # sampler
    cf_train_sampler = DistributedSampler(cf_train_dataset) if distributed else None
    cf_val_sampler = DistributedSampler(cf_val_dataset, drop_last=True, shuffle=False) if distributed else None

    # dataloader
    cf_train_loader = DataLoader(cf_train_dataset, batch_size=cf_batch_sz,
                                 shuffle=(cf_train_sampler is None), num_workers=workers,
                                 pin_memory=True, sampler=cf_train_sampler)
    cf_val_loader = DataLoader(cf_val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=workers, pin_memory=True, sampler=cf_val_sampler)

    return cf_train_loader, cf_val_loader, cf_train_sampler

def get_cue_conflict_dls(batch_size, workers):
    return DataLoader(CueConflict(), batch_size=batch_size, pin_memory=True, num_workers=workers)

def get_in9_dls(distributed, batch_size, workers, variations=['mixed_rand', 'mixed_same']):
    dls_in9 = {}
    for v in variations:
        in9_ds = Imagenet9(join('.', 'imagenet', 'data', 'in9', v))
        dls_in9[v] = in9_ds.make_loader(distributed=distributed,
                                        batch_size=batch_size,
                                        workers=workers)
    return dls_in9
