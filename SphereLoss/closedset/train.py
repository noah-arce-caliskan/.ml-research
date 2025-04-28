from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.benchmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np

import net_sphere

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from test import test

parser = argparse.ArgumentParser(description='PyTorch SphereFace')
parser.add_argument('--net', '-n', default='resnet', type=str)
parser.add_argument('--dataset', default="/home/noah/SphereLoss/known", type=str)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--optimizer', default='adamw', type=str)
parser.add_argument('--weights', default='/home/noah/SphereLoss/weights/current.pth', type=str)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def save_model(model, filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def train(epoch, args, train_loader):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    for inputs, targets in train_loader:
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        outputs_cos = outputs[0] # 0=cos_theta 1=phi_theta
        _, predicted = torch.max(outputs_cos.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()
    print('')
    acc = 100.0 * correct/total
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
    return acc

net = getattr(net_sphere, args.net)()
net.load_state_dict(torch.load('/home/noah/SphereLoss/weights/best_weights.pth'))
net.cuda()
criterion = net_sphere.AngleLoss()


if args.optimizer.lower() == 'sgd': optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=.0001)
elif args.optimizer.lower() == 'adam': optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=.0001)
elif args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.0001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((112, 96), padding=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
])

full_dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
train_loader = DataLoader(full_dataset, batch_size=args.bs, shuffle=True, num_workers=6)

print('Start: time={}'.format(dt()))
best_acc = 0.0
for epoch in range(args.epochs):
    net.train()
    criterion.train()
    train_acc = train(epoch, args, train_loader)
    scheduler.step()
    save_model(net, os.path.join('/home/noah/SphereLoss/weights', 'current.pth'))
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(net.state_dict(), os.path.join('/home/noah/SphereLoss/weights', 'best_weights.pth'))
    if epoch % 3 == 0:
        net.eval()
        criterion.eval()
        test(args)
        net.train()
        criterion.train()

print('Finish: time={}\n'.format(dt()))
