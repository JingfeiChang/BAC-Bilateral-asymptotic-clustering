import argparse
import os 
import pdb
parser = argparse.ArgumentParser(description='Attention Transfer Generatvie Adversarial pruning')

## Warm-up 
parser.add_argument(
    '--gpus',
    type=int,
    nargs='+',
    default=[0],
    help='Select gpu to use')
parser.add_argument(
    '--data_set',
    type=str,
    default='cifar10',
    help='Dataset to train: cifar10, cifar100 or imagenet')
parser.add_argument(
    '--data_path',
    type=str,
    default='./dataset',
    help='The directory where the input data is stored.')
parser.add_argument(
    '--job_dir',
    type=str,
    default='experiments/vgg16_cifar100',
    help='The directory where the summaries will be stored. vgg16, resnet56, googlenet')
parser.add_argument(
    '--prune_dir',
    type=str,
    default='experiments/vgg16_cifar10/checkpoint/model.pth',
    help='The directory where the summaries will be stored. vgg16, resnet56, googlenet')

parser.add_argument(
    '--reset',
    action='store_true',
    help='Reset the directory?')
parser.add_argument(
    '--resume', 
    type=str, 
    default=None,
    help='Load the model from the specified checkpoint.')
parser.add_argument(
    '--refine', 
    type=str, 
    default=None,
    help='Path to the model to be fine tuned.')

## Training
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_cifar',
    help='Architecture of model. default:resnet_cifar')
parser.add_argument(
    '--cfg',
    type=str,
    default='vgg16',
    help='Detail architecuture of model. default:vgg16')
parser.add_argument(
    '--num_epochs', 
    type=int,
    default=160,
    help='The num of epochs to train.')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,                     
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='Momentum for MomentumOptimizer.')
parser.add_argument(
    '--lr',
    type=float,
    default=1e-1
)
parser.add_argument(
    '--lr_decay_step',
    type=int,
    nargs='+', 
    default=[80, 120],
)
parser.add_argument(
    '--weight_decay', 
    type=float,
    default=2e-4,
    help='The weight decay of loss.')
parser.add_argument(
    '--beta', 
    type=float,
    default=0.98,
    help='The weight of previous retained channels.')
parser.add_argument(
    '--alpha', 
    type=float,
    default=0.5,
    help='The weight of weight pruning.')
parser.add_argument(
    '--random',
    action='store_true',
    help='Random weight initialize for finetune')
parser.add_argument(
    '--random_rule',
    type=str,
    default='l1_pretrain',
    help='Weight initialization criterion after random clipping. default:default optional:default,random_pretrain,l1_pretrain')
parser.add_argument(
    '--pruned',
    action='store_true',
    help='Load pruned model')

parser.add_argument(
    '--pretrained_model',
    type=str,
    default='./pretrained-cifar10/vgg16.pth.tar',
    help='Path to the model wait for Beepruning. default:None'
)

## Status
parser.add_argument(
    '--print_freq', 
    type=int,
    default=100,
    help='The frequency to print loss.')
parser.add_argument(
    '--test_only', 
    action='store_true',
    help='Test only?') 

args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

