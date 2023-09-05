import torch
import torch.nn as nn
import torch.optim as optim
from model.googlenet import Inception
from utils.options import args
from sklearn import preprocessing
from torch.autograd import Variable
from sklearn.cluster import DBSCAN, OPTICS
import utils.common as utils

import os
import copy
import math
import time
import random
import numpy as np
import heapq
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.transforms as transforms
from data import cifar10, cifar100
from importlib import import_module
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

checkpoint = utils.checkpoint(args)
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
logger = utils.get_logger(os.path.join(args.job_dir + 'logger.log'))
loss_func = nn.CrossEntropyLoss()

conv_num_cfg = {
    'vgg9': 6,
    'vgg16': 13,
    'vgg19': 16,
    'resnet56' : 27,
    'resnet110' : 54,
    'googlenet' : 27,
    'densenet':36,
    }

original_food_cfg = {
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'vgg19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'resnet56': [16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 
                 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'resnet110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                  32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                  64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'googlenet': [96, 16, 32, 128, 32, 96, 96, 16, 48, 112, 24, 64, 128, 24, 64, 144, 32, 64, 
                  160, 32, 128, 160, 32, 128, 192, 48, 128]
    }

reg_hook = [3, 5, 6, 10, 12, 13, 17, 19, 20, 24, 26, 27, 31, 33, 34, 
            38, 40, 41, 45, 47, 48, 52, 54, 55, 59, 61, 62] 

food_dimension = conv_num_cfg[args.cfg]
original_food = original_food_cfg[args.cfg]

def load_vgg_particle_model(model, random_rule, oristate_dict):
    #print(ckpt['state_dict'])
    #global oristate_dict
    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)-1
            orifilter_num1 = oriweight.size(1)
            currentfilter_num1 = curweight.size(1)-1
       

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                                
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                select_num = currentfilter_num1
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [0, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    
                    select_index.sort()
                for index_i, i in enumerate(select_index):
                    
                    state_dict[name + '.weight'][:, index_i, :, :] = \
                        oristate_dict[name + '.weight'][:,i, :, :]

                
                last_select_index = None

    model.load_state_dict(state_dict)

def load_google_particle_model(model, random_rule):
    global oristate_dict
    state_dict = model.state_dict()
    all_food_conv_name = []
    all_food_bn_name = []

    for name, module in model.named_modules():

        if isinstance(module, Inception):

            food_filter_channel_index = ['.branch5x5.3']  # the index of sketch filter and channel weight
            food_channel_index = ['.branch3x3.3', '.branch5x5.6']  # the index of sketch channel weight
            food_filter_index = ['.branch3x3.0', '.branch5x5.0']  # the index of sketch filter weight
            food_bn_index = ['.branch3x3.1', '.branch5x5.1', '.branch5x5.4'] #the index of sketch bn weight
            
            for bn_index in food_bn_index:
                all_food_bn_name.append(name + bn_index)

            for weight_index in food_filter_channel_index:
                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    #print(state_dict[conv_name].size())
                    #print(oristate_dict[conv_name].size())
                else:
                    select_index = range(orifilter_num)
         
            
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)


                select_index_1 = copy.deepcopy(select_index)


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                else:
                    select_index = range(orifilter_num)
                
                for index_i, i in enumerate(select_index):
                    for index_j, j in enumerate(select_index_1):
                            state_dict[conv_name][index_i][index_j] = \
                                oristate_dict[conv_name][i][j]



            for weight_index in food_channel_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)

                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]
                orifilter_num = oriweight.size(1)
                currentfilter_num = curweight.size(1)

                #print(state_dict[conv_name].size())
                #print(oristate_dict[conv_name].size())


                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()


                    for i in range(state_dict[conv_name].size(0)):
                        for index_j, j in enumerate(select_index):
                            state_dict[conv_name][i][index_j] = \
                                oristate_dict[conv_name][i][j]


            for weight_index in food_filter_index:

                conv_name = name + weight_index + '.weight'
                all_food_conv_name.append(name + weight_index)
                oriweight = oristate_dict[conv_name]
                curweight = state_dict[conv_name]

                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()

                    for index_i, i in enumerate(select_index):
                            state_dict[conv_name][index_i] = \
                                oristate_dict[conv_name][i]


    for name, module in model.named_modules(): #Reassign non sketch weights to the new network

        if isinstance(module, nn.Conv2d):

            if name not in all_food_conv_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']

        elif isinstance(module, nn.BatchNorm2d):

            if name not in all_food_bn_name:
                state_dict[name + '.weight'] = oristate_dict[name + '.weight']
                state_dict[name + '.bias'] = oristate_dict[name + '.bias']
                state_dict[name + '.running_mean'] = oristate_dict[name + '.running_mean']
                state_dict[name + '.running_var'] = oristate_dict[name + '.running_var']

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_resnet_particle_model(model, random_rule, oristate_dict):

    cfg = { 
           'resnet56': [9,9,9],
           'resnet110': [18,18,18],
           }

    state_dict = model.state_dict()
        
    current_cfg = cfg[args.cfg]
    last_select_index = None

    all_honey_conv_weight = []

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'
        for k in range(num):
            for l in range(2):
                conv_name = layer_name + str(k) + '.conv' + str(l+1)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                #logger.info('weight_num {}'.format(conv_weight_name))
                #logger.info('orifilter_num {}\tcurrentnum {}\n'.format(orifilter_num,currentfilter_num))
                #logger.info('orifilter  {}\tcurrent {}\n'.format(oristate_dict[conv_weight_name].size(),state_dict[conv_weight_name].size()))

                if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num-1), select_num)
                        select_index.sort()
                    else:
                        l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                        select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                        select_index.sort()
                    if last_select_index is not None:
                        #logger.info('last_select_index'.format(last_select_index))
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]  

                    last_select_index = select_index
                    #logger.info('last_select_index{}'.format(last_select_index)) 

                elif last_select_index != None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    #for param_tensor in state_dict:
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,state_dict[param_tensor].size()))
    #for param_tensor in model.state_dict():
        #logger.info('param_tensor {}\tType {}\n'.format(param_tensor,model.state_dict()[param_tensor].size()))
 

    model.load_state_dict(state_dict)


# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
    data_sets = CIFAR10
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)
    data_sets = CIFAR100
else:
    loader = imagenet.Data(args)
    data_sets = ImageFolder

fmap_block = []
def forward_hook(module, data_input, data_output):
    fmap_block.append(data_output)

# Training    
def train(model, optimizers, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    optimizer = optimizers[0]
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time
# Testing
def test(model, testLoader):
    
    model.eval()
    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()
    
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg




# Pruning

def FMPruning(model, trainLoader):
    print('==> Start FMPruning..')   
    netchannels=[]
    retain_c = []    
    a = 0
    # register hook        
    if args.arch == 'vgg_cifar':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):              
                a = a+1
                if a <= 13:               
                    handle = m.register_forward_hook(forward_hook)
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                
                #handle = m.register_forward_hook(forward_hook)
    elif args.arch == 'resnet_cifar':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                a = a+1
                if a % 2 == 0:
                    handle = m.register_forward_hook(forward_hook)  
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]) 
    elif args.arch == 'googlenet':
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                a = a+1
                if a in reg_hook:
                    handle = m.register_forward_hook(forward_hook)  
            transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                      
    trainset = data_sets(root=args.data_path, train=False, download=True, transform=transform_train)
    image = []
    for images, labels in trainset:
        images = Variable(torch.unsqueeze(images, dim=0).float(), requires_grad=False)
        image.append(images)
        #image = image.cuda()
    print(np.array(image[3]).shape)

    for i in random.sample(range(10000),200):
        imagetest = image[i].cuda()
        with torch.no_grad():
            model(imagetest)

    channels = conv_num_cfg[args.cfg]      
    netchannels = torch.zeros(channels)
    for s in range(channels):

        # change the size of fmap_block from (batchsize, channels, W, H) to (batchsize, channels, W*H)
        a, b, c, d = fmap_block[s].size()
        fmap_block[s] = fmap_block[s].view(a, b, -1)

        fmap_block[s] = torch.sum(fmap_block[s], dim=0)/a

        
        # clustering
        X = np.array(fmap_block[s].cpu())
        clustering = OPTICS(min_samples=5, metric='cosine', cluster_method='xi', xi=0.04).fit(X)
        
        # defult: eps=0.5, min_samples=5
        labels = clustering.labels_

        #print(labels)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
    
        netchannels[s] = netchannels[s]+n_clusters_+n_noise_

        #print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)
        retain_c.append(int(netchannels[s]))       

    handle.remove()  
    fmap_block.clear()
    print(retain_c) 
    return retain_c

def WTPruning(model):
    print('==> Start WTPruning..')
    retain_wc = []       
    i=0
    j = -1
    for m in model.modules():
        if args.arch == 'resnet_cifar':
            if isinstance(m, nn.Conv2d):
                i=i+1
                if i % 2 == 0:  # RN56
                    j = j + 1
                    kernels = m.weight
                    c_out, c_int, k_h, k_w = kernels.size()  
                    #print(kernels.shape)
                    kernels = kernels.view(c_out, c_int, -1)
                    #print(kernels.shape)
                    kernels = torch.sum(kernels, dim=1)/c_int
                    #print(kernels.shape)
                    kernels = kernels.detach().cpu().numpy()
                   
                    # clustering
                    X = np.array(kernels)
                    clustering = OPTICS(min_samples=3, metric='cosine', cluster_method='xi', xi=0.04).fit(X)
    
                    # defult: eps=0.5, min_samples=5
        
                    labels = clustering.labels_
    
                    #print(labels)
    
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    #print(n_clusters_)
                    n_noise_ = list(labels).count(-1)
                    #print(n_noise_)
            
                    retain_wc.append(int(n_clusters_+n_noise_)) 
        
                    #print('Estimated number of clusters: %d' % n_clusters_)
                    #print('Estimated number of noise points: %d' % n_noise_)
        elif args.arch == 'vgg_cifar':
            if isinstance(m, nn.Conv2d):
                j = j+1
                kernels = m.weight
                c_out, c_int, k_h, k_w = kernels.size()  
                #print(kernels.shape)
                kernels = kernels.view(c_out, c_int, -1)
                #print(kernels.shape)
                kernels = torch.sum(kernels, dim=1)/c_int
                #print(kernels.shape)
                kernels = kernels.detach().cpu().numpy()
               
                # clustering
                X = np.array(kernels)
                clustering = OPTICS(min_samples=5, metric='cosine', cluster_method='xi', xi=0.04).fit(X)
    
                # defult: eps=0.5, min_samples=5
                labels = clustering.labels_
    
                #print(labels)
    
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                #print(n_clusters_)
                n_noise_ = list(labels).count(-1)
                #print(n_noise_)
        
                retain_wc.append(int(n_clusters_+n_noise_)) 
                
        elif args.arch == 'googlenet':
            if isinstance(m, nn.Conv2d):
                i = i+1
                if i in reg_hook:
                    kernels = m.weight
                    c_out, c_int, k_h, k_w = kernels.size()  
                    #print(kernels.shape)
                    kernels = kernels.view(c_out, c_int, -1)
                    #print(kernels.shape)
                    kernels = torch.sum(kernels, dim=1)/c_int
                    #print(kernels.shape)
                    kernels = kernels.detach().cpu().numpy()
                    
                    # clustering
                    X = np.array(kernels)
                    clustering = OPTICS(min_samples=3, metric='cosine', cluster_method='xi', xi=0.04).fit(X)
        
                    # defult: eps=0.5, min_samples=5
                    
                    labels = clustering.labels_
        
                    #print(labels)
        
                    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                    #print(n_clusters_)
                    n_noise_ = list(labels).count(-1)
                    #print(n_noise_)
            
                    retain_wc.append(int(n_clusters_+n_noise_)) 
   
    print(retain_wc)
    return retain_wc
    
def main():
    start_epoch = 0
    best_acc = 0.0
    t = 1
    retain = original_food  #original channels config
    '''
    retain = []
    for i in range(food_dimension):
        retain.append(0)
    '''    
    print(retain)

    if args.arch == 'vgg_cifar':
        model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
        #model = import_module(f'model.{args.mmd}').vggmmd(args.cfg, foodsource=current_food).to(device)
    elif args.arch == 'resnet_cifar':
        model = import_module(f'model.{args.arch}').resnet(args.cfg, food=original_food).to(device)
    elif args.arch == 'googlenet':
        model = import_module(f'model.{args.arch}').googlenet(food=original_food).to(device) 
       
    print(model)    
    model_dict_s = model.state_dict()
    ckpt_o = torch.load(args.pretrained_model, map_location=device)
    state_dict_t = ckpt_o['state_dict']
    model_dict_s.update(state_dict_t)
    model.load_state_dict(model_dict_s)
    
    test(model, loader.testLoader)

    param_s = [param for name, param in model.named_parameters()]
    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    #scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
    scheduler_s = CosineAnnealingLR(optimizer_s, T_max=160)
    
    optimizers = [optimizer_s]
    schedulers = [scheduler_s]

    for epoch in range(start_epoch, args.num_epochs):

        if epoch > 0:        
            print('=> training')      
            train(model, optimizers, loader.trainLoader, args, epoch)      
            for s in schedulers:
                s.step()
                lr = s.get_last_lr()
                print(lr)       
            
            test_acc = test(model, loader.testLoader)
            
            print('**************************************************************************')
            
            is_best = best_acc < test_acc
            best_acc = max(test_acc, best_acc)
    
            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    
            state = {
                'state_dict': model_state_dict,
                'best_acc': best_acc,
                'optimizer_s': optimizer_s.state_dict(),
                'epoch': epoch + 1
            }
            checkpoint.save_model(state, is_best)    
        
        if epoch <= 20 and epoch % 10 == 0:           
            channels_p = []
            for i in range(food_dimension):
                channels_p.append(0)
            if epoch == 0:
                checkpt = torch.load(args.pretrained_model, map_location=device)  #state of the original model  
            else:
                checkpt = torch.load(args.prune_dir, map_location=device)  #state of the pruned model         
            
            retain_f = FMPruning(model, loader.trainLoader) 
            print(retain_f)
            retain_w = WTPruning(model)
            print(retain_w)
            for m in range(len(retain_f)):
                channels_p[m] = args.alpha*retain_w[m] + (1-args.alpha)*retain_f[m]
                #print(channels_p)
                if t == 1:
                    retain[m] = math.ceil(args.beta/math.sqrt(m+1)*retain[m] + (1-args.beta/math.sqrt(m+1))*channels_p[m])
                    if retain[m] < 5:
                        retain[m] = 5
                else:
                    retain[m] = math.ceil(args.beta/math.sqrt(m+1)*retain[m] + (1-args.beta/math.sqrt(m+1))*channels_p[m])
                    if retain[m] < 5:
                        retain[m] = 5
            t = t+1
            print('=> pruned channels')
            print(retain)
            
            if args.arch == 'vgg_cifar':
                model = import_module(f'model.{args.arch}').PSOVGG(args.cfg, foodsource=retain).to(device) 
                #model = import_module(f'model.{args.arch}').VGG(args.cfg).to(device)
                oristate_dict = checkpt['state_dict']
                load_vgg_particle_model(model, args.random_rule, oristate_dict)
            elif args.arch == 'resnet_cifar':
                model = import_module(f'model.{args.arch}').resnet(args.cfg,food=retain).to(device)
                oristate_dict = checkpt['state_dict']
                load_resnet_particle_model(model, args.random_rule, oristate_dict) 
            elif args.arch == 'googlenet':
                model = import_module(f'model.{args.arch}').googlenet(food=retain).to(device)  

            epoch = checkpt['epoch']
            
            param_s = [param for name, param in model.named_parameters()]
            optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            #scheduler_s = MultiStepLR(optimizer_s, milestones=args.lr_decay_step, gamma=0.1)
            scheduler_s = CosineAnnealingLR(optimizer_s, T_max=10)
            
            optimizers = [optimizer_s]
            schedulers = [scheduler_s]      
        
    #print(model)    
    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))



if __name__ == '__main__':
    main()
