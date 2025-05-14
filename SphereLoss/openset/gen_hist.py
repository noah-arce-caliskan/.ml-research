import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import losses_nets


def histogram(net, batch_size, data_dir='/home/noah/SphereLoss/openset/test', output_path='/home/noah/SphereLoss/openset/histograms/hist.png', device=None, bins=50, title="OPENSET: Embedding Similarity Distribution"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device).eval()

    data_transforms = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    net.eval()
    net.feature = True

    all_embs, all_labels = [], []
    with torch.no_grad():
        for imgs, labs in data_loader:
            imgs = imgs.to(device)
            outs = net(imgs)
            all_embs.append(outs.cpu())
            all_labels.append(labs)

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    sim_matrix = all_embs @ all_embs.T

    eq = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
    mask_same = torch.triu(eq, diagonal=1)
    mask_diff = torch.triu(~eq, diagonal=1)

    same_sims = sim_matrix[mask_same].numpy()
    diff_sims = sim_matrix[mask_diff].numpy()

    plt.figure(figsize=(10,6))
    plt.hist(same_sims, bins=bins, alpha=0.5, label=f"Same (μ={same_sims.mean():.3f}, σ={same_sims.std():.3f})", density=True)
    plt.hist(diff_sims, bins=bins, alpha=0.5, label=f"Diff (μ={diff_sims.mean():.3f}, σ={diff_sims.std():.3f})", density=True)
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    net = net_sphere.resnet(num_classes=731, feature_dim=512, s=64, m=4).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    net.load_state_dict(torch.load('/home/noah/SphereLoss/openset/weights/best_weights.pth'))
    histogram(net, 256)
