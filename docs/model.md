# Model

A compact CNN suitable for CIFAR-10 (32x32 RGB):

- 3x blocks: Conv -> ReLU -> MaxPool
- Flatten -> FC(512) -> FC(10)

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
```

Defaults:

 
- Optimizer: SGD(lr=1e-3, momentum=0.9)
- Loss: CrossEntropyLoss
- Scheduler: StepLR(step_size=15, gamma=0.5)
- Early stopping: patience=7, min_delta=0.001


Weights:

 
- Best checkpoint saved to `cifar10_best_model.pth`
