from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import models
from utils.flops import *
import sys
import utils.common as myutils
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--data_path', type=str, default='./cifar-10')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=260, metavar='N',
                    help='number of epochs to train (default: 260)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='/checkpoint/VGG', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='VGG', type=str,
                    help='architecture to use')
parser.add_argument('--sr',default=0.00001,type=float)
parser.add_argument('--threshold',default=0.01,type=float)
parser.add_argument('--lameda',default=0.000001,type=float)
parser.add_argument('--a',default=0.4,type=float)
parser.add_argument('--use_function', default='piecewise', choices=['piecewise','continuous'],type=str,help='piecewise or continuous')

args = parser.parse_args()
print(args)
checkpoint = myutils.checkpoint(args)
print_logger = myutils.get_logger(os.path.join(args.save, "run/logger.log"))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.num_classes == 10:
    train_set = datasets.CIFAR10(args.data_path, train=True)
    test_set = datasets.CIFAR10(args.data_path, train=False)
else:
    train_set = datasets.CIFAR100(args.data_path, train=True)
    test_set = datasets.CIFAR100(args.data_path, train=False)

train_set.transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    transforms.RandomErasing(value=[0.4914, 0.4822, 0.4465]),
])
test_set.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
model = models.__dict__[args.arch](num_classes=args.num_classes)
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
history_score = np.zeros((args.epochs + 1, 3))


def train(epoch,dic_threshold):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss_stripe_var, loss_filter_var , loss_mean = criterion_shape(model,dic_threshold)
        loss_new = loss + args.a*args.lameda*loss_stripe_var + (1-args.a)*args.lameda*loss_filter_var
        loss_new.backward()
        ###########learning the shape of filter with filter skeleton################
        if args.sr and args.threshold:
            model.update_skeleton(args.sr, dic_threshold)
        ############################################################################
        optimizer.step()
        if batch_idx % 100 == 0:
            print_logger.info(
                'Train Epoch: {0} [{1}/{2}]:\t'
                'Loss: {loss:.6f}\t'.format(
                epoch, batch_idx * len(data),len(train_loader.dataset),loss=loss_new.item()))
    history_score[epoch][0] = avg_loss / len(train_loader) #loss
    history_score[epoch][1] = train_acc / float(len(train_loader)) #acc


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print_logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n")

    return correct / float(len(test_loader.dataset))

tool=models.vgg.tool(args.use_function)


dic_threshold=tool.creat_dic(model, args.threshold)
criterion_shape=models.vgg.cal_shape(dic_threshold)
###############################
best_prec1 = 0.
best_threshold = dic_threshold
for epoch in range(args.epochs):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

    if epoch>=args.epochs*0.5 and epoch%10==0:
        dic_threshold=tool.change_rate(model,args.threshold,dic_threshold)
        print_logger.info(dic_threshold)
        args.sr = 0

    train(epoch,dic_threshold)
    prec1 = test()
    history_score[epoch][2] = prec1
    np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')
    if epoch>=args.epochs*0.5 and prec1 > best_prec1:
        best_prec1 = prec1
        best_threshold = dic_threshold
        torch.save(model.state_dict(), os.path.join(args.save, 'best.pth.tar'))

print_logger.info('Best accuracy:{best:.4f}'.format(best=best_prec1))
print_logger.info(best_threshold)
history_score[-1][0] = best_prec1
np.savetxt(os.path.join(args.save, 'train_record.txt'), history_score, fmt='%10.5f', delimiter=',')
##############pruning filter in filter without finetuning#################
print_logger.info('-------------------------------------')
print_logger.info('Loading model ......')
model.load_state_dict(torch.load(os.path.join(args.save, 'best.pth.tar')))
test()
model.prune(best_threshold)
torch.save(model.state_dict(), os.path.join(args.save, 'pruned.pth.tar'))
print_model_param_nums(model)
count_modelvgg_param_flops(model)
print_logger.info('-------------------------------------')
#########################################################




