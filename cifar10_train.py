# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Step 1: Load CIFAR-10 dataset
    from torch.utils.data import random_split, DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load training and test datasets
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split train into train/validation
    val_size = 5000
    train_size = len(full_trainset) - val_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Step 2: Define the Convolutional Neural Network (CNN)
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Convolutional Layer 1
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Convolutional Layer 2
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Convolutional Layer 3
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer

            # Fully Connected Layers
            self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Flattened output from convolution layers
            self.fc2 = nn.Linear(512, 10)  # Output layer (10 classes)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
            x = self.pool(torch.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
            x = self.pool(torch.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool
            x = x.view(-1, 128 * 4 * 4)  # Flatten the output
            x = torch.relu(self.fc1(x))  # Fully Connected Layer 1
            x = self.fc2(x)  # Output Layer
            return x

    # Instantiate the network
    net = Net()

    # Step 3: Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Optimizer

    # Step 4: Training the Model (with validation, early stopping, scheduler, and plots)
    epochs = 50
    patience = 7  # for early stopping
    min_delta = 0.001
    best_val_acc = 0.0
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    best_model_state = None

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        net.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.3f}, Train Acc={train_acc:.3f}, Val Loss={val_loss:.3f}, Val Acc={val_acc:.3f}")

        # Early stopping
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = net.state_dict()
            torch.save(best_model_state, 'cifar10_best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print('Finished Training')
    print(f'Best Val Accuracy: {best_val_acc*100:.2f}%')

    # Step 5: Test the Model on Test Data (with best model)
    net.load_state_dict(torch.load('cifar10_best_model.pth'))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    # Step 6: Plot loss and accuracy curves
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.tight_layout()
    plt.show()
