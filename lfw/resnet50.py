from torchvision import datasets, transforms
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

full_training_set = datasets.ImageFolder(root='/workspace/noah/lfw/lfw_training', transform=transform)

total_size = len(full_training_set)
split1_size = int(0.6 * total_size)
split2_size = total_size - split1_size

split1, split2 = random_split(full_training_set, [split1_size, split2_size])

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = 30
    batch_size = 32
    learning_rate = 0.0001
    num_workers = 16

    test_dir = '/workspace/noah/lfw/lfw_testing'

    train_loader = DataLoader(split1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(full_training_set.classes)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = 0.0
    best_loss = float('inf')
    best_model_path = '/workspace/noah/lfw/best_model.pth'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / split1_size
        epoch_acc = running_corrects.double() / split1_size

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%')

        if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
            best_acc = epoch_acc
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    model.eval()
    test_running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_running_corrects += torch.sum(preds == labels.data)

    test_acc = test_running_corrects.double() / len(test_dataset)
    print(f'Test Accuracy: {test_acc*100:.2f}%')

if __name__ == '__main__':
    main()
