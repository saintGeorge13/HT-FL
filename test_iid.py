import pandas as pd
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
import argparse

import torchvision.models as MODELS
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.optimizer import Optimizer, required
import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Run Multi-Level Local SGD.')
    parser.add_argument('--data', type=str, nargs='?', default=0,
                            help='dataset to use in training.')
    parser.add_argument('--model', type=str, nargs='?', default=0,
                            help='model to use in training.')
    parser.add_argument('--hubs', type=int, nargs='?', default=1,
                            help='number of hubs in system.')
    parser.add_argument('--workers', type=int, nargs='?', default=1,
                            help='number of workers per hub.')
    parser.add_argument('--tau', type=int, nargs='?', default=1,
                            help='number of local iterations for worker.')
    parser.add_argument('--q', type=int, nargs='?', default=1,
                            help='number of sub-network iterations before global averaging.')
    parser.add_argument('--graph', type=int, nargs='?', default=0,
                            help='graph file ID to use for hub network.')
    parser.add_argument('--epochs', type=int, nargs='?', default=1000,
                            help='Number of epochs/global iterations to train for.')
    parser.add_argument('--batch', type=int, nargs='?', default=1000,
                            help='Batch size to use in Mini-batch SGD.')
    parser.add_argument('--prob', type=int, nargs='?', default=0,
                            help='Indicates with probability distribution to use for workers.')
    parser.add_argument('--fed', type=str2bool, nargs='?', default=False,
                            help='IID or not')
    parser.add_argument('--chance', type=float, nargs='?', default=0.55,
                            help='Fixed probability of taking gradient step.')
    parser.add_argument('--percentage', type=float, nargs='?', default=1,
                            help='Fixed probability of taking gradient step.')
    parser.add_argument('--non_iid', type=int, default=0,
                        help='How to be non-iid 0:unequal| 1:equal|2:MixUp|3:Dirichlet')
    parser.add_argument('--num_class', type=int, nargs='?', default=2,
                            help='How many classes for equal.')
    parser.add_argument('--uniform', type=float, nargs='?', default=0.1,
                            help='How many uniform distribution for Mixup')
    parser.add_argument('--dir', type=float, nargs='?', default=0.3,
                            help='Dirichlet level')
    args = parser.parse_args()
    print(args)
    return args
def get_dataset(dataset, args, num_users):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'cifar':
        train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                   ]))
        test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                   ]))        

    else:
        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
        test_dataset = datasets.MNIST('./data', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

        # sample training data amongst users
    if args.fed:
        user_groups = IID(train_dataset, num_users)
    else:
        if args.non_iid == 0:
            user_groups = noniid_unequal(train_dataset, num_users)
        if args.non_iid == 1:
            user_groups = noniid(train_dataset, num_users, args.num_class)
        if args.non_iid == 2:
            user_groups = Mix_up(train_dataset, num_users, args.uniform)
        if args.non_iid == 3:
            user_groups = Dirichlet(train_dataset, num_users, args.dir)
    return train_dataset, test_dataset, user_groups
def Dirichlet(dataset, num_users, dir):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = np.array(dataset.targets)
    n_cls = (int(torch.max(torch.tensor(label)))) + 1
    n_data = data.shape[0]
    dir_level = dir

    cls_priors = np.random.dirichlet(alpha=[dir_level] * n_cls, size=num_users)
    # cls_priors_init = cls_priors # Used for verification
    prior_cumsum = np.cumsum(cls_priors, axis=1)
    idx_list = [np.where(label == i)[0] for i in range(n_cls)]
    cls_amount = [len(idx_list[i]) for i in range(n_cls)]
    idx_worker = [[None] for i in range(num_users)]

    for curr_worker in range(num_users):
        for data_sample in range(n_data // num_users):
            curr_prior = prior_cumsum[curr_worker]
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            while cls_amount[cls_label] <= 0:
                # If you run out of samples
                correction = [[1 - cls_priors[i, cls_label]] * n_cls for i in range(num_users)]
                cls_priors = cls_priors / correction
                cls_priors[:, cls_label] = [0] * num_users
                curr_prior = np.cumsum(cls_priors, axis=1)
                cls_label = np.argmax(np.random.uniform() <= curr_prior)

            cls_amount[cls_label] -= 1
            if idx_worker[curr_worker] == [None]:
                idx_worker[curr_worker] = [idx_list[cls_label][0]]
            else:
                idx_worker[curr_worker] = idx_worker[curr_worker] + [idx_list[cls_label][0]]

            idx_list[cls_label] = idx_list[cls_label][1::]
    data_list = [idx_worker[curr_worker] for curr_worker in range(num_users)]
    return data_list
def Mix_up(dataset, num_users, uniform):
    data = dataset.data
    data = data.numpy() if torch.is_tensor(data) is True else data
    label = dataset.targets
    n_workers = num_users
    homo_ratio = uniform
    
    n_data = data.shape[0]
    n_homo_data = int(n_data * homo_ratio)
    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo = np.array(range(0, n_homo_data))
        data_homo_list= np.split(data_homo, n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        # label_hetero_sorted, index = torch.sort(torch.tensor(label_hetero))
        idxs_labels = np.vstack((np.arange(n_homo_data, n_data), label_hetero))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        data_hetero_list = np.split(idxs, n_workers)
    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                        zip(data_homo_list, data_hetero_list)]

    return data_list
def IID(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def noniid(dataset, num_users, num_class):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = num_users * num_class, int(dataset.data.shape[0] / num_users / num_class)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int)
    return dict_users
def noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = int(dataset.data.shape[0] / 50), 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 4
    max_shard = 20

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    for i in range(num_users):
        dict_users[i] = dict_users[i].astype(int)
    return dict_users
start_time = time.time()
args = parse_args()
hubs = args.hubs
dataset = args.data
q = args.q
tau = args.tau
workers = args.workers
epochs = args.epochs 
BATCH_SIZE = args.batch
prob = args.prob
fed = args.fed
chance = args.chance
train_dataset, test_dataset, user_groups = get_dataset(dataset, args, workers*hubs)
all = []
for i in range(100):
    x = []
    for k in range(len(list(user_groups[i]))):
        x.append(train_dataset[list(user_groups[i])[k]][1])
    all.append(x)

import scipy.stats
from scipy.stats import ks_2samp # 小于0.05，分布不一样
sum = 0
for i in range(100):
    for j in range(i + 1, 100):
        ks = ks_2samp(all[i], all[j])
        # KL = scipy.stats.entropy(all[i], all[j]) 
        if ks.pvalue >= 0.05:
            sum += 1
print("KS分布一样: ", sum)     
  
  
from scipy.stats import ttest_ind, f
def ftest(s1,s2):
    '''F检验样本总体方差是否相等'''
    F = np.var(s1)/np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2*abs(0.5-f.cdf(F,v1,v2))
    if p_val < 0.05:
        equal_var=False
    else:
        equal_var=True
    return equal_var
	 	
def ttest_ind_fun(s1,s2):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(s1,s2)
    ttest,pval = ttest_ind(s1,s2,equal_var=equal_var)
    if pval >= 0.05 and equal_var:	
        return True
    else:
        return False
sum = 0
for i in range(100):
    for j in range(i + 1, 100):
        ks = ttest_ind_fun(all[i], all[j])
        if ks:
            sum += 1
print("T分布一样:", sum)

for i in range(100):
    for j in range(i + 1, 100):
        KL = scipy.stats.entropy(all[i], all[j])
        print(KL)

      