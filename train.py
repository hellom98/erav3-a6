from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import os
import random
import numpy as np

# Set seeds for reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call seed_everything() before creating model and loading data
seed_everything(1)

class Net(nn.Module):
    def create_crbdp_block(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1, normalize=False, dropout=False, pool=False):
        layers = []
        # Conv layer
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, bias=False))
        # Optional Batch Normalization
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels, affine=False))
        # ReLU activation
        layers.append(nn.ReLU())
        # Optional Dropout
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        # Optional MaxPool
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def __init__(self):
        super(Net, self).__init__()
        self.block1 = self.create_crbdp_block(1, 16, padding=0, normalize=True, dropout=False, pool=False) # 26x26
        self.conv1 = nn.Conv2d(16, 8, 1, bias=False)
        self.block2 = self.create_crbdp_block(8, 16, padding=0, normalize=True, dropout=False, pool=True)  # 12x12
        self.conv2 = nn.Conv2d(16, 8, 1, bias=False)
        self.block3 = self.create_crbdp_block(8, 16, padding=0, normalize=True, dropout=False, pool=False)  # 10x10
        self.conv3 = nn.Conv2d(16, 8, 1, bias=False)
        self.block4 = self.create_crbdp_block(8, 16, padding=0, normalize=True, dropout=False, pool=True)  # 4x4
        self.conv4 = nn.Conv2d(16, 8, 1, bias=False)
        self.block5 = self.create_crbdp_block(8, 32, padding=0, normalize=True, dropout=False, pool=True)  # 1x1
        self.conv5 = nn.Conv2d(32, 10, 1, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.conv1(x)
        x = self.block2(x)
        x = self.conv2(x)
        x = self.block3(x)
        x = self.conv3(x)
        x = self.block4(x)
        x = self.conv4(x)
        x = self.block5(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 1, 28, 28))

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                        transforms.RandomRotation(degrees=10),
                        transforms.ToTensor(),
                        transforms.RandomApply([transforms.Lambda(lambda x: transforms.functional.erase(x, i=random.randint(0, 5), j=random.randint(0, 5), h=5, w=5, v=0))], p=0.5),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy for current batch
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(target)
        pbar.set_description(desc= f'epoch={epoch} accuracy={accuracy:.2f}% loss={loss.item():.4f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def show_misclassified_images(model, device, test_loader, epoch, num_images=5):
    model.eval()
    misclassified = {i: [] for i in range(10)}  # Dictionary to store misclassified images by true label
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Find misclassified images
            incorrect_mask = pred.ne(target)
            for idx in range(len(target)):
                if incorrect_mask[idx]:
                    true_label = target[idx].item()
                    pred_label = pred[idx].item()
                    if len(misclassified[true_label]) < num_images:
                        img = data[idx].cpu().squeeze()
                        misclassified[true_label].append((img, pred_label))

            # Check if we have enough images for all digits
            if all(len(images) >= num_images for images in misclassified.values()):
                break
    
    # Create plot
    fig, axes = plt.subplots(10, num_images, figsize=(15, 25))
    fig.suptitle('Misclassified Images by True Label', fontsize=16)
    
    for true_label in range(10):
        for j, (img, pred_label) in enumerate(misclassified[true_label]):
            axes[true_label, j].imshow(img, cmap='gray')
            axes[true_label, j].axis('off')
            axes[true_label, j].set_title(f'Pred: {pred_label}')
        axes[true_label, 0].set_ylabel(f'True: {true_label}', rotation=0, labelpad=40)
    
    plt.tight_layout()
    plt.savefig(f'misclassified_samples_{epoch}.png')
    plt.close()

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

# Modify the training loop to run fewer epochs in CI environment
num_epochs = 6 if os.environ.get('CI') else 20

# Replace the final training loop with:
for epoch in range(1, num_epochs + 1):
    current_lr = scheduler.get_last_lr()[0]
    print(f'\nEpoch {epoch}, Learning Rate: {current_lr:.0e}')
    
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    if not os.environ.get('CI'):  # Skip saving images in CI environment
        show_misclassified_images(model, device, test_loader, epoch)
    scheduler.step()

# Save the model at the end
torch.save(model.state_dict(), 'model.pth')