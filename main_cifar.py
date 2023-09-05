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

