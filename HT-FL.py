import pandas as pd
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
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
import Net.tiny
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
def load_graph(g,num_centers):
    """
    Load in adjacency matrix from graph file with ID 'g'
    """
    A = []
    graph_file = open("graphs/graph_"+str(g)+".txt",'r')
    lines = graph_file.readlines()
    i = 0
    for line in lines:
        row = line.split(' ')
        A.append([])
        for r in row:
            A[i].append(int(r))
        i += 1

    return A
def get_model(model):
    if model == "cnn":
        import Net.MnistNet
        model = Net.MnistNet.MnistNet()
    if model == "resnet":
        import Net.Resnet
        model = Net.Resnet.ResNet18()
    if model == "resnet34":
        import Net.Resnet_new
        model = Net.Resnet_new.ResNet34()
    elif model == "densenet":
        import Net.Densenet
        model = Net.Densenet.DenseNet121()
    elif model == "resnext":
        import Net.ResneXt
        model = Net.ResneXt.ResNeXt29_2x64d()
    elif model == "vgg":
        import Net.VGG
        model = Net.VGG.VGG('VGG19')
    elif model == "mobilenet":
        import Net.MobileNetV2
        model = Net.MobileNetV2.MobileNetV2()
    elif model == "effecientnet":
        import Net.EffecientNet
        model = Net.EffecientNet.EfficientNetB0()
    elif model == "googlenet":
        import Net.GoogleNet
        model = Net.GoogleNet.GoogLeNet()
    elif model == "shufflenet":
        import Net.ShuffleNetV2
        model = Net.ShuffleNetV2.ShuffleNetV2(1)
    elif model == "regnet":
        import Net.RegNet
        model = Net.RegNet.RegNetY_400MF()
    elif model == "dpn":
        import Net.DPN
        model = Net.DPN.DPN92()
    return model
def get_client_epochs(local_epochs, prob, n):
    """
    Random decisions to take a local step or not
    """
    client_epochs = local_epochs
    if prob == 1:
        client_epochs = 0
        for p in range(0,local_epochs):
            client_epochs += random.random() < 0.5
    if prob == 2:
        client_epochs = 0
        for p in range(0,local_epochs):
            client_epochs += random.random() < n + 0.05
    if prob == 3:
        client_epochs = 0
        if n < 0.2:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.1 
        else:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.6 
    if prob == 4:
        client_epochs = 0
        if n < 0.5:
            client_epochs = local_epochs 
        else:
            for p in range(0,local_epochs):
                client_epochs += random.random() < 0.5
    if prob == 5:
        client_epochs = 0
        for p in range(0,local_epochs):
            client_epochs += random.random() < 0.75
    return client_epochs
def train(train_loader, model, optimizer, epochs=1):   
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.0001)
    best_model_wts = copy.deepcopy(model).state_dict()
    best_loss = 10000
    params = []
    for param in model.parameters():
        params.append(0)
    for e in range(epochs):
        for t,(x,y) in enumerate(train_loader):
            model.train()   # set model to training mode
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i = 0
            for param in model.parameters():
                params[i] += param.grad.data
                i += 1
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_wts = copy.deepcopy(model).state_dict()
    model.load_state_dict(best_model_wts)
    return model
def average_model(model, client_models, fracs):
    """
    Average Neural Net with other client models
    """
    num_clients = len(client_models)
    params = []
    for param in model.parameters():
        params.append(0)
    j = 0
    for net in client_models:
        i = 0
        for param in net.parameters():
            params[i] += param.data*fracs[j] 
            i += 1
        j += 1
    i = 0
    for param in model.parameters():
        param.data = params[i]#/num_clients 
        i += 1
    return model
def inference(model, test_loader):
    # model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()
    # model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    tmp = [0] * 10
    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = F.cross_entropy(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        
    accuracy = correct/total
    return loss/batch_idx, accuracy
def main():
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
    if args.graph != 5:
        A = load_graph(args.graph,hubs)
    else:
        A = np.ones((hubs,hubs), dtype=int)

    model = get_model(args.model)

    
    # model = MODELS.resnet18(pretrained=True)
    models = []
    for n in range(0, hubs):
        models.append(copy.deepcopy(model)) #hub之间不同的方法
        # models.append(model)
    avg_model = model
    train_dataset, test_dataset, user_groups = get_dataset(dataset, args, workers*hubs)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)
    train_loaders = []

    offset_hub = []
    for n in range(0, hubs):
        sum = 0
        for i in range(0, workers):
            train_loader = DataLoader(Partition(train_dataset, list(user_groups[n*workers+i])),
                                batch_size=BATCH_SIZE, shuffle=True)
            train_loaders.append(train_loader)
            sum += len(list(user_groups[n*workers+i]))
        offset_hub.append(sum)
    
    distribution = []
    for i in range(hubs):
        for j in range(workers):
            if args.model == "resnet34":
                x = [0]*200
            else:
                x = [0]*10
            for k in range(len(list(user_groups[workers*i+j]))):
                x[int(train_dataset[list(user_groups[workers*i+j])[k]][1])] += 1
            distribution.append(x)
    KL_div = []
    if args.model == "resnet34":
        norm =  [0.005] * 200
    else:
        norm =  [0.1] * 10
    import scipy.stats
    for i in range(hubs*workers):
        KL_div.append(round(scipy.stats.entropy(distribution[i], norm), 4))
    KL_div = np.array(KL_div)
    print("KL of all workers: ", np.mean(np.array(KL_div)))
    distribution = np.array(distribution)
    order = []
    for i in range(hubs):
        kl = np.sort(KL_div[i*workers:(i+1)*workers])
        kl_place = list(np.argsort(-KL_div[i*workers:(i+1)*workers]))
        order_hub = []
        tmp_list = [kl_place[0]]
        tmp_distribution = distribution[kl_place[0] + workers*i]
        kl_place.remove(kl_place[0])
        while len(kl_place) != 0:
            flag = False
            now_best_list = 0
            now_best_distribution = tmp_distribution
            for j in range(len(kl_place)):
                if scipy.stats.entropy(tmp_distribution + distribution[kl_place[j] + workers*i], norm) < scipy.stats.entropy(now_best_distribution, norm):
                    now_best_distribution = tmp_distribution + distribution[kl_place[j] + workers*i]
                    now_best_list = kl_place[j]
                    flag = True
            if flag:     
                tmp_list.append(now_best_list)
                tmp_distribution = now_best_distribution
                kl_place.remove(now_best_list)
                    
                    
            if not flag or len(tmp_list) >= 3:
                order_hub.append(tmp_list)
                if len(kl_place) != 0:
                    tmp_list = [kl_place[0]]
                    tmp_distribution = distribution[kl_place[0] + workers*i]
                    kl_place.remove(kl_place[0])
                else:
                    tmp_list = []
        if len(tmp_list) != 0:
            order_hub.append(tmp_list)
        order.append(order_hub)
    tmp_KL = []
    for i in range(hubs):
        for j in range(len(order[i])):
            tmp_distribution = distribution[workers * i + order[i][j][0]]
            for k in range(1, len(order[i][j])):
                tmp_distribution = tmp_distribution + distribution[workers * i + order[i][j][k]]
            tmp_KL.append(round(scipy.stats.entropy(tmp_distribution, norm), 4))
    print("KL in sequence; ", np.mean(np.array(tmp_KL)))
    offset_worker = []
    for n in range(0, hubs):
        offset_worker_temp = []
        for i in range(len(order[n])): 
            sum = 0
            for j in range(len(order[n][i])):
                train_loader = DataLoader(Partition(train_dataset, list(user_groups[order[n][i][j]+n*workers])),
                                    batch_size=BATCH_SIZE, shuffle=True)
                sum += len(list(user_groups[order[n][i][j]+n*workers]))
            offset_worker_temp.append(sum)
        offset_worker.append(offset_worker_temp)
    linear_order = []
    for i in range(hubs):
        kl = np.sort(KL_div[i*workers:(i+1)*workers])
        kl_place = list(np.argsort(-KL_div[i*workers:(i+1)*workers]))
        linear_order.append(kl_place)
    import math
    g = 2e6
    beita_1 = 1e-3
    beita_2 = 4
    yibusilou = 1.38e-23
    M =  42.63 * 1024 * 1024 * 8 # 82.73, 0.08
    t = 8 # 15, 0.016
    P_st = 3
    _k = 0.2
    _Q = 0.05
    P_sum_wtw = 0
    P_sum_wth = 0
    P_sum_wtw_min = 0
    P_dy_list = []
    P_tr_list = []
    for n in range(0, hubs):   
        distance_matrix = np.random.rand(workers+1,workers+1) * 100
        for i in range(workers + 1):
            distance_matrix[i][i] = 0
        for i in range(1, workers + 1):
            for j in range(i):
                distance_matrix[i][j] = distance_matrix[j][i]
        P_dy = 0
        for i in range(workers):
            util = np.random.rand(1) / 2 + 0.5
            if util < 0.7:
                P_dy += util * _k
                P_dy_list.append(util * _k)
            else:
                P_dy += (0.7 * _k + _Q * (util - 0.7) ** 2)
                P_dy_list.append(0.7 * _k + _Q * (util - 0.7) ** 2)

        P_tran = 0
        for j in range(len(order[n])):
            distance = distance_matrix[0][order[n][j][0]+1]
            P_tran += (g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1))
            distance = distance_matrix[order[n][j][len(order[n][j])-1]+1][0]
            P_tran += (g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1))
            P_tr_list.append(g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1))
            for k in range(1, len(order[n][j])):
                distance = distance_matrix[order[n][j][k-1]+1][order[n][j][k]+1]
                P_tran += (g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1))
                P_tr_list.append(g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1))
        P_sum_wtw += (P_st * workers + P_dy + P_tran)

        P_tran = 0
        for i in range(1, workers + 1):
            distance = distance_matrix[0][i]
            P_tran += (g * yibusilou / beita_1 * math.pow(distance, beita_2) * (math.pow(2, M / g / t) - 1)) * 2
        P_sum_wth += (P_st * workers + P_dy + P_tran)
        
        print("worker找worker", P_sum_wtw)
        print("worker找hub", P_sum_wth)
    P_sum = []
    for i in range(hubs * workers):
        P_sum.append(P_dy_list[i] + P_tr_list[i] + 3)
    print("总能耗：", np.mean(np.array(P_sum)))
    fedname = ""
    if fed:
        fedname = "fed"
    costs = np.zeros(epochs)
    success_rate = np.zeros(epochs)

    learning_rate = 0.01
    # gr = []
    # lo = []
    # AVG_MODEL = model
    for e in tqdm(range(0, epochs*q)):
        if e > epochs*q/3:
            learning_rate = 0.01 
        if e > epochs*2*q/3:
            learning_rate = 0.01
        # gradient = []
        # loss = []
        for n in range(0, hubs):
            models[n] = models[n].to(device)
            # MODEL = copy.deepcopy(models[n])
            optimizer = optim.SGD(models[n].parameters(), lr=learning_rate, momentum=0.5, weight_decay=0.0001)# True最好,momentum0.9 hub之间不同的方法
            
            client_models = []
            for i in range(len(order[n])): 
                temp_models = copy.deepcopy(models[n])         
                for j in range(len(order[n][i])):
                    client_epochs = get_client_epochs(tau, prob, order[n][i][j]/workers)
                    train_loader = train_loaders[order[n][i][j]+n*workers]
                    model = train(train_loader, temp_models, optimizer, client_epochs)
                client_models.append(temp_models)
            models[n] = average_model(models[n], client_models, np.array(offset_worker[n])/np.sum(offset_worker[n]))

            # for i in range(0, workers):
            #     client_epochs = get_client_epochs(tau, prob, i/workers)
            #     # client_epochs = 10
            #     clients_run[e, n, i] = client_epochs
                
            #     train_loader = train_loaders[n*workers+i]
            #     # train_loader = train_loaders[linear_order[n][i]+n*workers]
                
            #     # optimizer.setperiod( int(train_loader.sampler.num_samples / BATCH_SIZE + 1) * client_epochs )
            #     # optimizer.setglobal(copy.deepcopy(models[n]))
            #     model = train(train_loader, models[n], optimizer, client_epochs)
            
        #     MODEL = MODEL.to(device)
        #     total_norm = 0
        #     for p, P in zip(models[n].parameters(), MODEL.parameters()):
        #         param_norm = (p.data - P.data).norm(2)
        #         total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** (1. / 2)
        #     gradient.append(total_norm)
        
        # cost, rate = inference(models[n], test_loader)
        # loss.append(cost)      
        
        # gr.append(round(np.mean(np.array(gradient)), 4))
        # lo.append(round(np.mean(np.array(loss)), 4))
        if e % q == 0:
            tmp = []
            for i in range(0, hubs):
                neighbor_models = [models[j] for j in range(hubs) if(A[i][j]==1)]
                offset_neighbor = [offset_hub[j] for j in range(hubs) if(A[i][j]==1)]
                tmp.append(average_model(copy.deepcopy(models[i]), neighbor_models, 
                        np.array(offset_neighbor)/np.sum(offset_neighbor))) 
            for i in range(0, hubs):
                models[i] = tmp[i]
            
            avg_model = avg_model.to(device)  
            avg_model = average_model(avg_model, models, np.array(offset_hub)/np.sum(offset_hub))
            costs[int(e/q)], success_rate[int(e/q)] = inference(avg_model, test_loader)
            print("Weighted Average Model Loss: "+str(costs[int(e/q)]))
            print("Success_rate: ", success_rate[int(e/q)])
            
            # if e != 0:
            #     total_norm = 0
            #     for p, P in zip(avg_model.parameters(), AVG_MODEL.parameters()):
            #         param_norm = (p.data - P.data).norm(2)
            #         total_norm += param_norm.item() ** 2
            #     total_norm = round(total_norm ** (1. / 2), 4)
            #     gr.append(total_norm)
            # AVG_MODEL = copy.deepcopy(avg_model)
            filename = f"results/loss_model{args.model}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}_per{args.percentage}_noniid{args.non_iid}_num_class{args.num_class}_uniform{args.uniform}_dir{args.dir}.p"
            pickle.dump(costs, open(filename,'wb'))
            filename = f"results/accuracy_model{args.model}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}_per{args.percentage}_noniid{args.non_iid}_num_class{args.num_class}_uniform{args.uniform}_dir{args.dir}.p"
            pickle.dump(success_rate, open(filename,'wb'))
            # filename = f"results/target_model{args.model}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}_per{args.percentage}_noniid{args.non_iid}_num_class{args.num_class}_uniform{args.uniform}_dir{args.dir}.p"
            # pickle.dump(target, open(filename,'wb'))
    #     print(gr)
    #     print(lo)
    # lo_dif = []
    # for i in range(len(gr)-1):
    #     tmp = abs(lo[i] - lo[i+1])
    #     lo_dif.append(round(tmp / gr[i+1], 4))
    # filename = f"results/lo_dif_model{args.model}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}_per{args.percentage}_noniid{args.non_iid}_num_class{args.num_class}_uniform{args.uniform}_dir{args.dir}.txt"  
    # with open(filename,"w") as f:
    #     f.write(str(lo_dif))  
    # filename = f"results/lo_model{args.model}_data{dataset}_hubs{hubs}_workers{workers}_tau{tau}_q{q}_graph{args.graph}_prob{prob}{fedname}_per{args.percentage}_noniid{args.non_iid}_num_class{args.num_class}_uniform{args.uniform}_dir{args.dir}.txt"  
    # with open(filename,"w") as f:
    #     f.write(str(lo)) 
class Partition(Dataset):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]
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
    elif dataset == 'tiny-imagenet':
        train_dataset = Net.tiny.TinyImageNetDataset(
                        root='./data/tiny-imagenet', split='train', download=False,
                        transform=transforms.Compose([
                                    transforms.RandomRotation(10),  # Add RandomRotation
                                    transforms.RandomCrop(64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
                                ]))
        test_dataset = Net.tiny.TinyImageNetDataset(
                            root='./data/tiny-imagenet', split='test', download=False,
                            transform=transforms.Compose([
                                        transforms.RandomRotation(10),  # Add RandomRotation
                                        transforms.RandomCrop(64, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
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

        data_hetero_list = np.array(np.split(idxs, n_workers))
        data_hetero_list = np.random.permutation(data_hetero_list)
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


if __name__ == "__main__":
    main()
