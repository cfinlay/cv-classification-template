"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Training template for DNN computer vision research in PyTorch')
parser.add_argument('--datadir', type=str, required=True, metavar='DIR',
        help='data storage directory')
parser.add_argument('--dataset', type=str,help='dataset (default: "cifar10")',
        default='cifar10', metavar='DS',
        choices=['cifar10','cifar100', 'TinyImageNet','Fashion','mnist'])
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
        help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--logdir', type=str, default=None,metavar='DIR',
        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')
parser.add_argument('--seed', type=int, default=None, metavar='S',
        help='random seed (default: int(time.time()) )')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet34',
        help='Model architecture (default: ResNet34)')
group1.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--cutout',type=int, default=0, metavar='N',
        help = 'Cutout size, if data loader supports cutout (default: 0)')
group1.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
group1.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
group1.set_defaults(bn=True)
group1.add_argument('--last-layer-nonlinear', 
        action='store_true', default=False)
group1.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
group1.add_argument('--no-bias',action='store_false', dest='bias',
       help = "Don't use biases")
group1.set_defaults(bias=False)
group1.add_argument('--kernel-size',type=int, default=3, metavar='K',
        help='convolution kernel size (default: 3)')
group1.add_argument('--model-args',type=str, 
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')
group1.add_argument('--greyscale',action='store_true', dest='greyscale',
        help = "Make images greyscale")
group1.set_defaults(greyscale=False)


group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]', help='List of epochs and multiplier '
        'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.008]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
       help='SGD momentum parameter (default: 0.9)')


group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--decay',type=float, default=5e-4, metavar='L',
        help='Lagrange multiplier for weight decay (sum '
        'parameters squared) (default: 5e-4)')
