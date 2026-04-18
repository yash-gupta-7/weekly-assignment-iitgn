import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

def get_model(num_classes, mode='feature_extraction'):
    """
    Returns a pre-trained ResNet-18 model configured for the task.
    """
    model = models.resnet18(pretrained=True)
    
    if mode == 'feature_extraction':
        for param in model.parameters():
            param.requires_grad = False
    elif mode == 'fine_tuning':
        # Freeze only the first few layers
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    
    # Replace classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cpu'):
    model.to(device)
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            # Filter out unlabeled (-1)
            mask = labels != -1
            if not mask.any(): continue
            images, labels = images[mask].to(device), labels[mask].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                mask = labels != -1
                if not mask.any(): continue
                images, labels = images[mask].to(device), labels[mask].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total if total > 0 else 0
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_acc'].append(acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {acc:.2f}%")
        
    return model, history
