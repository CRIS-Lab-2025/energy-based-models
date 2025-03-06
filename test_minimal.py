import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Import our custom modules
from util.config import Config

# Create a custom config for convolutional network
config = Config()
config.device = "cpu"  # Change to "cuda" if you have a GPU

# Set up a simple convolutional network
class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.fc(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use a small subset of MNIST for testing
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)
train_dataset = torch.utils.data.Subset(train_dataset, range(1000))

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)
test_dataset = torch.utils.data.Subset(test_dataset, range(200))

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=False
)

# Initialize the network
model = SimpleConvNet()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}, Accuracy: {100 * correct/total:.2f}%')
                running_loss = 0.0
        
        print(f'Epoch [{epoch+1}], Accuracy: {100 * correct/total:.2f}%')

# Testing function
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct/total:.2f}%')
    return 100 * correct/total

# Run training
print("Starting training...")
train(model, train_loader, criterion, optimizer, num_epochs=1)

# Run testing
print("Testing the model...")
test_accuracy = test(model, test_loader)

print("Training and testing complete!")
print(f"Final test accuracy: {test_accuracy:.2f}%") 