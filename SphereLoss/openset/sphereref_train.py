# add git readme

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import losses_nets
from sklearn.metrics import roc_curve, auc
from gen_hist import histogram
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--hist_path', type=str, default='/home/noah/SphereLoss/openset/histograms/hist.png')
parser.add_argument('--train_path', type=str, default='/home/noah/SphereLoss/openset/train')
parser.add_argument('--val_path', type=str, default="/home/noah/SphereLoss/openset/val")
parser.add_argument('--test_path', type=str, default='/home/noah/SphereLoss/openset/test')
parser.add_argument('--best_weights', type=str, default='/home/noah/SphereLoss/openset/weights/best_weights.pth')

parser.add_argument('--refine_epochs', type=int, default=50)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--refine_lr', type=float, default=.0001)
parser.add_argument('--ref_best_weights', type=str, default='/home/noah/SphereLoss/openset/weights/ref_best_weights.pth')
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

train_ds= datasets.ImageFolder(args.train_path, transform=train_transform)
val_ds = datasets.ImageFolder(args.val_path, transform=eval_transform)
test_ds = datasets.ImageFolder(args.test_path, transform=eval_transform)
train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=6)
val_loader= DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=6)
test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=6)

net = losses_nets.resnet(num_classes=len(train_ds.classes), feature_dim=512, s=80, m=4).to(device)

net.feature = False
criterion = losses_nets.AngleLoss()

if args.optimizer.lower() == 'sgd': optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=.0001)
elif args.optimizer.lower() == 'adam': optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=.0001)
elif args.optimizer.lower() == 'adamw': optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=5)


def val_metrics(embeddings, labels):
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

best_eer = 1.0
no_improve = 0
patience = 10 #stop early

for epoch in range(args.epochs):
    net.train()
    net.feature = False
    criterion.train()
    total_loss = 0
    for imgs, lbls in train_loader: 
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()
        cos_phi = net(imgs)
        loss = criterion(cos_phi, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # val embds
    net.eval()
    net.feature = True
    emb_val = []
    lbl_val = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            emb = net(imgs.to(device))
            emb_val.append(emb.cpu())
            lbl_val.append(lbls)
    emb_val = torch.cat(emb_val)
    lbl_val = torch.cat(lbl_val)
    val_auc, val_eer = val_metrics(emb_val, lbl_val)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.3f} Val AUC: {val_auc:.3f} Val EER: {val_eer:.3f}")

    scheduler.step(avg_loss)

    if val_eer < best_eer:
        best_eer = val_eer
        torch.save(net.state_dict(), args.best_weights)
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print('Not improving. Stopping. ')
            break

def pairwise_distances(x):
    x_norm = (x**2).sum(1).view(-1,1)
    dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    return torch.sqrt(torch.clamp(dist, min=1e-12))


net.load_state_dict(torch.load(args.best_weights))
net.train(); net.feature = True

for p in net.parameters(): p.requires_grad=False
for p in net.backbone.layer3.parameters(): p.requires_grad=True
for p in net.backbone.layer4.parameters(): p.requires_grad=True
for p in net.fc.parameters(): p.requires_grad=True
for p in net.layer_norm.parameters(): p.requires_grad=True

ref_net = losses_nets.RefinementNet(input_dim=512, hidden_dim=1024, output_dim=512).to(device)
triplet_crit = nn.TripletMarginLoss(margin=0.1)
angle_crit  = losses_nets.AngleLoss()
opt_params = [{'params':net.backbone.layer3.parameters(),'lr': args.refine_lr*0.1},
    {'params':net.backbone.layer4.parameters(),'lr': args.refine_lr*0.1},
    {'params':net.fc.parameters(),'lr': args.refine_lr},
    {'params':net.layer_norm.parameters(),'lr': args.refine_lr},
    {'params':ref_net.parameters(),'lr': args.refine_lr}]
    
ref_opt = optim.AdamW(opt_params, weight_decay=1e-4)
scheduler_ref = optim.lr_scheduler.ReduceLROnPlateau(ref_opt, mode='min', factor=0.5, patience=4)

best_eer_ref = 1.0
no_imp_ref   = 0

for ep in range(args.refine_epochs):
    net.train()
    ref_net.train()
    total_loss_r = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        emb0 = net(imgs)
        emb1 = ref_net(emb0)

        dists = pairwise_distances(emb1)
        anchors, positives, negatives = [], [], []
        for i in range(len(lbls)):
            pos_mask = (lbls==lbls[i])
            pos_mask[i]=False
            neg_mask = (lbls!=lbls[i])
            if pos_mask.sum()>0 and neg_mask.sum()>0:
                hardest_pos = emb1[pos_mask][torch.argmax(dists[i][pos_mask])]
                hardest_neg = emb1[neg_mask][torch.argmin(dists[i][neg_mask])]
                anchors.append(emb1[i])
                positives.append(hardest_pos)
                negatives.append(hardest_neg)
        if len(anchors)==0: continue
        anc = torch.stack(anchors)
        pos = torch.stack(positives)
        neg = torch.stack(negatives)
        loss_tri = triplet_crit(anc, pos, neg)

        cos_phi_ref = net.angle_linear(F.normalize(emb1))

        loss_ang = angle_crit(cos_phi_ref, lbls)
        loss_r = loss_tri + 0.1 * loss_ang
        
        ref_opt.zero_grad()
        loss_r.backward()
        ref_opt.step()
        total_loss_r += loss_r.item()

    avg_r = total_loss_r / len(train_loader)

    net.eval()
    ref_net.eval()
    emb_val, lbl_val = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            emb_val.append(ref_net(net(imgs.to(device))).cpu())
            lbl_val.append(lbls)
    emb_val = torch.cat(emb_val)
    lbl_val = torch.cat(lbl_val)
    auc_ref, eer_ref = val_metrics(emb_val, lbl_val)
    print(f"Ref Ep{ep+1} Loss:{avg_r:.3f} AUC:{auc_ref:.3f} EER:{eer_ref:.3f}")

    scheduler_ref.step(avg_r)
    if eer_ref < best_eer_ref:
        best_eer_ref = eer_ref
        torch.save(ref_net.state_dict(), args.ref_best_weights)
        no_imp_ref = 0
    else:
        no_imp_ref += 1
        if no_imp_ref >= 10:
            break

net.eval()
ref_net.eval()
use_ref = (best_eer_ref < best_eer)
if use_ref:
    ref_net.load_state_dict(torch.load(args.ref_best_weights))
    model = nn.Sequential(net, ref_net)
else:
    model = net

all_e, all_l = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        emb0 = net(imgs.to(device))
        emb1 = ref_net(emb0) if use_ref else emb0
        all_e.append(emb1.cpu())
        all_l.append(lbls)
all_e = torch.cat(all_e); all_l = torch.cat(all_l)

histogram(model, args.bs)
auc_t, eer_t = val_metrics(all_e, all_l)
print(f"Final Test AUC: {auc_t:.4f} EER: {eer_t:.4f}")
