
import torch
import argparse
import net_sphere
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def test(data_dir, net, criterion, batch_size=256, use_cuda=False):

    test_transforms = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    net.eval()
    criterion.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            cos_theta = outputs[0]
            _, preds = cos_theta.max(dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    print(f"[Test Loss: {avg_loss:.4f}] [Test Acc: {accuracy:.2f}%]")
    return avg_loss, accuracy