from __future__ import print_function
import numpy as np
import torch
import contextlib

import os
import sys
import errno
import numpy as np
from PIL import Image
import torch.utils.data as data
import contextlib
import pickle
# from datasets.base import *
import copy
import imageio
import numpy as np
import os
from torchvision import datasets, transforms

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm

import PIL.Image
from PIL import Image

__all__ = ['TinyImageNetDataset']




@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import torchvision.datasets.accimage as accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def build_set(root, split, imgs, noise_type='pairflip', noise_rate=0.5):
    """
       Function to return the lists of paths with the corresponding labels for the images
    Args:
        root (string): Root directory of dataset
        split (str): ['train', 'gallery', 'query'] returns the list pertaining to training images and labels, else otherwise
    Returns:
        return_list: list of 236_comb_fromZeroNoise-tuples with 1st location specifying path and 2nd location specifying the class
    """

    tmp_imgs = imgs

    argidx = np.argsort(tmp_imgs)




def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img

class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                                root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                            wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {'train': [],  # [img_path, id, nid, box]
                        'val': [],  # [img_path, id, nid, box]
                        'test': []  # img_path
                        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))

class TinyImageNetDataset(data.Dataset):
    def __init__(self, root, split='train', preload=False, load_transform=None,
                transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root, download)
        self.split = split
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()
        self.loader = default_loader

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        if split == 'test':
            self.samples = tinp.paths['val']
        else:
            self.samples = tinp.paths['train']

        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(split)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                               dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if split != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])
        else:
            self.data = np.array([i[0] for i in self.samples])
            self.targets = np.array([i[1] for i in self.samples])
            self.classes = set(self.targets)
            self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.split == 'test' else self.label_data[idx]
        else:
            img, target = self.data[idx], self.targets[idx]
            #img = imageio.imread(img)
            img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target