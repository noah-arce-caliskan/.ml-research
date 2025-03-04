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

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

label_map = {} # mapping folders to num labels

parser = argparse.ArgumentParser(description='PyTorch SphereFace')
parser.add_argument('--net', '-n', default='sphere20a', type=str)
parser.add_argument('--dataset', default='/home/noah/SphereLoss/lfw_testing', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='')
parser.add_argument('--epochs', default=10, type=int)
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# got rid of alignment bc, is there a way I can incorporate it??

def dataset_load(name,filename,pindex,cacheobj,dummy):
    global label_map 
    folder = os.path.basename(os.path.dirname(filename)) # get label from parent foldername
    if folder not in label_map: label_map[folder] = len(label_map)
    classid = label_map[folder]

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    if ':train' in name:
        if random.random()>0.5: img = cv2.flip(img,1)
        if random.random()>0.5:
            rx = random.randint(0,2*2)
            ry = random.randint(0,2*2)
            img = img[ry:ry+112,rx:rx+96,:]
        else:
            img = img[2:2+112,2:2+96,:]
    else:
        img = img[2:2+112,2:2+96,:]


    img = img.transpose(2, 0, 1).reshape((3,112,96))
    img = ( img - 127.5 ) / 128.0
    label = np.zeros((1,1), np.float32)
    label[0,0] = classid
    return (img, label)

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

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

        printoneline(dt(), 'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f, lamb=%.2f, it=%d'
                     % (epoch, train_loss/(batch_idx+1), 100.0*correct/total, correct, total,
                        loss_val, criterion.lamb, criterion.it))
        batch_idx += 1
    print('')


net = getattr(net_sphere, args.net)()
# net.load_state_dict(torch.load('sphere20a_0.pth'))
net.cuda()
criterion = net_sphere.AngleLoss()

transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 255 - 127.5) / 128.0)
])

train_dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=6)

print('start: time={}'.format(dt()))
for epoch in range(args.epochs):
    if epoch in [0, 10, 15, 18]:
        if epoch!=0: args.lr *= 0.1
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    train(epoch, args, train_loader)
    save_model(net, '{}_{}.pth'.format(args.net, epoch))
print('finish: time={}\n'.format(dt()))
