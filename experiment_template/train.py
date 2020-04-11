import random
import time, datetime
import os, shutil, sys
import yaml
import ast, bisect
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torchnet as tnt

sys.path.append('../')
import dataloader
from dataloader import cutout
import cvmodels as models

# -------------
# Initial setup
# -------------

# Parse command line arguments 
from argparser import parser
args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# Set random seed
if args.seed is None:
    args.seed = int(time.time())
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Set and create logging directory
if args.logdir is None:
    args.logdir = os.path.join('./logs/',args.dataset,args.model,
            '{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))
os.makedirs(args.logdir, exist_ok=True)



# Print arguments to std out
# and save argument values to yaml file,
# so we know exactly how this experiment ran,
# and so we can re-load the model later
print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

args_file_path = os.path.join(args.logdir, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)


# Data loaders
workers=4
test_loader = getattr(dataloader, args.dataset)(args.datadir,
        mode='test', transform=False,
        batch_size=args.test_batch_size,
        greyscale=args.greyscale,
        num_workers=workers,
        shuffle=False,
        pin_memory=has_cuda)

image_shape = test_loader.image_shape
transforms = [cutout(args.cutout,channels=image_shape[0] )]
train_loader = getattr(dataloader, args.dataset)(args.datadir,
        mode='train', transform=True,
        greyscale=args.greyscale,
        batch_size=args.batch_size,
        training_transforms = transforms,
        num_workers=workers,
        shuffle=True,
        pin_memory=has_cuda,
        drop_last=True)

# Initialize model
classes = train_loader.classes
model_args = ast.literal_eval(args.model_args)
in_channels = 3 if not args.greyscale else 1
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=args.kernel_size, 
                  in_channels=in_channels,
                  softmax=False,last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
if args.dataset in ['cifar10','cifar100','Fashion']:
    model = getattr(models.cifar, args.model)(**model_args)
elif args.dataset=='TinyImageNet':
    model = getattr(models.tinyimagenet, args.model)(**model_args)
elif args.dataset=='mnist':
    model = getattr(models.mnist, args.model)(**model_args)

print('\n')
print(model)
print('\n')


# Loss function and regularizers
criterion = nn.CrossEntropyLoss()


# Move to GPU if available

if has_cuda:
    criterion = criterion.cuda(0)
    model = model.cuda(0)
    if torch.cuda.device_count()>1:
        pmodel = nn.DataParallel(model)
    else:
        pmodel = model
else:
    pmodel = model


# ------------------------------------
# Optimizer and learning rate schedule
# ------------------------------------
bparams=[]
oparams=[]
for name, p in pmodel.named_parameters():
    if 'bias' in name:
        bparams.append(p)
    else:
        oparams.append(p)

# Only layer weight matrices should have weight decay, not layer biases
optimizer = optim.SGD([{'params':oparams,'weight_decay':args.decay},
                       {'params':bparams,'weight_decay':0.}],
                  lr = args.lr,
                  momentum = args.momentum,
                  nesterov = False)

def scheduler(optimizer,args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler
schedule = scheduler(optimizer,args)



# --------
# Training
# --------
decay = args.decay # penalize by the sum of parameters squared


trainlog = os.path.join(args.logdir,'training.csv')
traincolumns = ['index','time','loss']
with open(trainlog,'w') as f:
    logger = csv.DictWriter(f, traincolumns)
    logger.writeheader()

ix=0 #count of gradient steps

def train(epoch, ttot):
    global ix

    # Put the model in train mode (turn on dropout, unfreeze
    # batch norm parameters)
    pmodel.train()

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()

    with open(trainlog,'a') as f:
        logger = csv.DictWriter(f, traincolumns)

        for batch_ix, (data, target) in enumerate(train_loader):

            if has_cuda:
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            output = pmodel(data)
            loss = criterion(output, target)
            loss.backward()

            optimizer.step()

            if np.isnan(loss.data.item()):
                raise ValueError('model returned nan during training')

            t = ttot + time.perf_counter() - tepoch
            fmt = '{:.4f}'
            logger.writerow({'index':ix, 
                'time': fmt.format(t),
                'loss': fmt.format(loss.item())})

            if (batch_ix % args.log_interval == 0 and batch_ix > 0):
                print('[Epoch %2d, batch %3d] penalized training loss: %.3g' %
                    (epoch, batch_ix, loss.data.item()))
            ix +=1

    if has_cuda:
        torch.cuda.synchronize()

    return ttot + time.perf_counter() - tepoch


# ------------------
# Evaluate test data
# ------------------
testlog = os.path.join(args.logdir,'test.csv')
testcolumns = ['epoch','time','fval','pct_err','train_fval','train_pct_err']
with open(testlog,'w') as f:
    logger = csv.DictWriter(f, testcolumns)
    logger.writeheader()

def test(epoch, ttot):
    pmodel.eval()

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = pmodel(data)


            top1_train.add(output.data, target.data)
            loss = criterion(output, target)
            train_loss.add(loss.data.item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        # Evaluate test data
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = pmodel(data)

            loss = criterion(output, target)

            top1.add(output, target)
            test_loss.add(loss.item())

        t1 = top1.value()[0]
        l = test_loss.value()[0]

    # Report results
    with open(testlog,'a') as f:
        logger = csv.DictWriter(f, testcolumns)
        fmt = '{:.4f}'
        logger.writerow({'epoch':epoch,
            'fval':fmt.format(l),
            'pct_err':fmt.format(t1),
            'train_fval':fmt.format(lt), 
            'train_pct_err':fmt.format(t1t),
            'time':fmt.format(ttot)})

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
            %(epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
            %('training loss',lt,t1t))

    return test_loss.value()[0], top1.value()[0]




# -------------------------------
# Now cook for 2 hours at 350 F
# -------------------------------
def main():


    save_model_path = os.path.join(args.logdir, 'checkpoint.pth.tar')
    best_model_path = os.path.join(args.logdir, 'best.pth.tar')

    pct_max = 100.*(1 - 1.0/classes)
    fail_max=5
    fail_count = fail_max
    time = 0.
    pct0 = 100.
    for e in range(args.epochs):


        time = train(e, time)

        loss, pct_err= test(e,time)
        if pct_err >= pct_max:
            fail_count -= 1

        torch.save({'ix': ix,
                    'epoch': e + 1,
                    'model': args.model,
                    'state_dict':model.state_dict(),
                    'pct_err': pct_err,
                    'loss': loss
                    }, save_model_path)
        if pct_err < pct0:
            shutil.copyfile(save_model_path, best_model_path)
            pct0 = pct_err

        if fail_count < 1:
            raise ValueError('Percent error has not decreased in %d epochs'%fail_max)

        # Update the learning rate
        schedule.step()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
