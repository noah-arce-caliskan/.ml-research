# train on refinement loss
# why not put on eval mode? i saw threads saying it gets rid of dropout and puts batchnorm to an average mean/stddev instead of by batch
#edit feature dims / s / m
from __future__ import print_function
import os, argparse, datetime, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import net_sphere
from sklearn.metrics import roc_curve, auc

def metrics(embeddings, labels):
    sim = embeddings @ embeddings.T
    eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_pos = torch.triu(eq, 1)
    mask_neg = torch.triu(~eq, 1)
    same = sim[mask_pos].cpu().numpy()
    diff = sim[mask_neg].cpu().numpy()
    
    y_true = np.concatenate([np.ones_like(same), np.zeros_like(diff)])
    y_score = np.concatenate([same, diff])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return roc_auc, eer


def make_histogram(embeddings, labels):
    sim = embeddings @ embeddings.T
    eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask_same = torch.triu(eq, 1)
    mask_diff = torch.triu(~eq, 1)
    same = sim[mask_same].cpu().numpy()
    diff = sim[mask_diff].cpu().numpy()
    plt.figure(figsize=(8,5))
    plt.hist(same, bins=50, alpha=0.5, density=True, label=f'Same (μ={same.mean():.3f}, σ={same.std():.3f})')
    plt.hist(diff, bins=50, alpha=0.5, density=True, label=f'Diff (μ={diff.mean():.3f}, σ={diff.std():.3f})')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('/home/noah/SphereLoss/openset/histograms/hist.png')
    plt.close()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--bs', type=int,   default=256)
parser.add_argument('--optimizer', type=str,   default='adamw')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((112,112), padding=10),
    transforms.ColorJitter(0.5,0.5,0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    transforms.RandomErasing(p=0.4, scale=(0.02,0.4), ratio=(0.3,3.3))
])
eval_transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

train_ds = datasets.ImageFolder('/home/noah/SphereLoss/openset/known', transform=train_transform)
val_ds   = datasets.ImageFolder('/home/noah/SphereLoss/openset/val',   transform=eval_transform)
test_ds  = datasets.ImageFolder('/home/noah/SphereLoss/openset/test',  transform=eval_transform)
train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=6)
val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=6)
test_loader  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=6)

num_classes = len(train_ds.classes)
net = net_sphere.resnet(num_classes=num_classes, feature_dim=512, s=64, m=4).to(device)
net.feature = False

criterion = net_sphere.AngleLoss()
# head_params = (list(net.fc.parameters())+list(net.layer_norm.parameters())+list(net.dropout.parameters())+list(net.angle_linear.parameters()))
# backbone_params = list(net.backbone.parameters())
# optimizer = optim.AdamW([{'params': head_params,'lr': 1e-3},{'params': backbone_params,'lr': 1e-5},],weight_decay=1e-4)
if args.optimizer.lower() == 'sgd': optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=.0001)
elif args.optimizer.lower() == 'adam': optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=.0001)
elif args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3,patience=3)

best_eer = 1.0
no_improve = 0
patience = 10 #stop early

best_path = '/home/noah/SphereLoss/openset/weights/best_weights.pth'

for epoch in range(100):
    net.train()
    net.feature = False
    criterion.train()
    total_loss = 0
    for imgs, lbls in train_loader: 
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        cos_phi = net(imgs)
        loss = criterion(cos_phi, lbls)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # val embds
    net.eval()
    net.feature = True
    emb_val, lbl_val = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            emb = net(imgs.to(device))
            emb_val.append(emb.cpu())
            lbl_val.append(lbls)
    emb_val = torch.cat(emb_val)
    lbl_val = torch.cat(lbl_val)
    val_auc, val_eer = metrics(emb_val, lbl_val)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.3f} | Val EER: {val_eer:.3f}")

    scheduler.step(avg_loss)

    # save best and early stop
    if val_eer < best_eer:
        best_eer = val_eer
        torch.save(net.state_dict(), best_path)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"No improvement in {patience} epochs—stopping.")
            break

net.load_state_dict(torch.load(best_path))
net.eval()
net.feature = True
emb_test, lbl_test = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        emb = net(imgs.to(device))
        emb_test.append(emb.cpu())
        lbl_test.append(lbls)
emb_test = torch.cat(emb_test)
lbl_test = torch.cat(lbl_test)


os.makedirs('/home/noah/SphereLoss/openset/histograms', exist_ok=True)
make_histogram(emb_test, lbl_test)
print("Best weights and histogram saved.")