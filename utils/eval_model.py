import argparse
import os, sys

import numpy as np
import pandas as pd
import pickle as pk
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import grad
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torch.utils.data import Subset

from flashlight.utils.loaders import get_model





parser = argparse.ArgumentParser('Gathers stats of a pretrained model, and saves results to a dataframe.')

parser.add_argument('--data-dir', type=str,required=True,
        metavar='DIR', 
        help='Directory where data is saved')
parser.add_argument('--dataset', type=str, choices=['cifar100','cifar10','mnist'],required=True)
parser.add_argument('--model-dir', type=str, required=True,metavar='DIR',
        help='Directory where model is saved')
parser.add_argument('--num-images', type=int, default=1000,metavar='N',
        help='total number of images to attack (default: 1000)')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time')

parser.add_argument('--pth-name', type=str, default='best.pth.tar',
        help='name of PyTorch save file to load')
parser.add_argument('--strict', action='store_true', dest='strict',
        help='only allow exact matches to model keys during loading (default)')
parser.add_argument('--no-strict', action='store_false', dest='strict',
        help='allow inexact matches to model keys durign loading')
parser.set_defaults(strict=True)

args = parser.parse_args()

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

# Data  and model loading code
# ----------------------------
transform = transforms.Compose([transforms.ToTensor()])

if args.dataset=='cifar100':
    classes = Nc = 100
    root = os.path.join(args.data_dir,'cifar100/')
    ds = CIFAR100(root, download=True, train=False, transform=transform)
elif args.dataset=='cifar10':
    classes = Nc = 10
    root = os.path.join(args.data_dir,'cifar10/')
    ds = CIFAR10(root, download=True, train=False, transform=transform)
elif args.dataset=='mnist':
    classes = Nc = 10
    root = os.path.join(args.data_dir,'mnist')
    ds = MNIST(root, download=True, train=False, transform=transform)

ix = torch.arange(args.num_images)
subset = Subset(ds, ix)

loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

model = get_model(args.model_dir, classes, pth_name=args.pth_name, 
        strict=args.strict, has_cuda=has_cuda)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

# Real work begins
# ----------------

Nsamples = args.num_images

criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        

Loss = torch.zeros(Nsamples).cuda()
NormGradLoss = torch.zeros(Nsamples).cuda()
Top1 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
Rank = torch.zeros(Nsamples,dtype=torch.int64).cuda()
Top5 = torch.zeros(Nsamples,dtype=torch.uint8).cuda()
NegLogPmax = torch.zeros(Nsamples).cuda()
NegLogP5 = torch.zeros(Nsamples).cuda()



sys.stdout.write('\nRunning through dataloader:\n')
Jx = torch.arange(Nc).cuda().view(1,-1)
Jx = Jx.expand(args.batch_size, Nc)
for i, (x,y) in enumerate(loader):
    sys.stdout.write('  Completed [%6.2f%%]\r'%(100*i*args.batch_size/Nsamples))
    sys.stdout.flush()

    x, y = x.cuda(), y.cuda()

    x.requires_grad_(True)

    yhat = model(x)
    p = yhat.softmax(dim=-1)

    psort , jsort = p.sort(dim=-1,descending=True)
    b = jsort==y.view(-1,1)
    rank = Jx[b]
    pmax = psort[:,0]
    logpmax = pmax.log()

    p5,ix5 = psort[:,0:5], jsort[:,0:5]
    ix1 =  jsort[:,0]
    sump5 = p5.sum(dim=-1)

    loss = criterion(yhat, y)
    g = grad(loss.sum(),x)[0]
    gn = g.view(len(y),-1).norm(dim=-1)





    top1 = ix1==y
    top5 = (ix5==y.view(args.batch_size,1)).sum(dim=-1)

    ix = torch.arange(i*args.batch_size, (i+1)*args.batch_size,device=x.device)

    Loss[ix] = loss.detach()
    Rank[ix]= rank.detach()
    Top1[ix] = top1.detach()
    Top5[ix] = top5.detach().type(torch.uint8)
    NegLogPmax[ix] = -logpmax.detach()
    NegLogP5[ix] = -sump5.log().detach()
    NormGradLoss[ix] = gn.detach()
sys.stdout.write('   Completed [%6.2f%%]\r'%(100.))

df = pd.DataFrame({'loss':Loss.cpu().numpy(),
                   'top1':np.array(Top1.cpu().numpy(),dtype=np.bool),
                   'top5':np.array(Top5.cpu().numpy(), dtype=np.bool),
                   'neg_log_pmax': NegLogPmax.cpu().numpy(),
                   'neg_log_p5': NegLogP5.cpu().numpy(),
                   'norm_grad_loss':NormGradLoss.cpu().numpy(),
                   'rank': Rank.cpu().numpy()})


print('\n\ntop1 error: %.2f%%,\ttop5 error: %.2f%%'%(100-df['top1'].sum()/Nsamples*100, 100-df['top5'].sum()/Nsamples*100))

ix1 = np.array(df['top1'], dtype=bool)
ix5 = np.array(df['top5'], dtype=bool)
ix15 = np.logical_or(ix5,ix1)
ixw = np.logical_not(np.logical_or(ix1, ix5))

df['type'] = pd.DataFrame(ix1.astype(np.int8) + ix5.astype(np.int8))
d = {0:'mis-classified',1:'top5',2:'top1'}
df['type'] = df['type'].map(d)
df['type'] = df['type'].astype('category')

df.to_pickle(os.path.join(args.model_dir,'eval_model_%s.pkl'%args.loss))
#
##if __name__=='__main__':
##    main()
