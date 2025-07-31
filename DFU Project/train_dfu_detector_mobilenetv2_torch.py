import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Custom DFU Dataset (Albumentations)
# ------------------------------
class DFUDataset(Dataset):
    def __init__(self, image_folder, class_transforms):
        self.dataset = datasets.ImageFolder(image_folder)
        self.class_transforms = class_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = self.class_transforms[label]
        augmented = transform(image=image)
        image = augmented['image']
        return image, label

# ------------------------------
# Data Augmentation
# ------------------------------
image_size = 224

majority_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

minority_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.7),
    A.Rotate(limit=40, p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(),
    ToTensorV2()
])

# ------------------------------
# Paths and Class Mappings
# ------------------------------
train_dir = 'DFU/Patches/wagner_classification/train'
val_dir = 'DFU/Patches/wagner_classification/val'
class_names = os.listdir(train_dir)
num_classes = len(class_names)
print("Classes:", class_names)

# Adjust which are minority classes if needed
class_transforms = {}
for idx in range(num_classes):
    if idx in [2, 3, 4]:
        class_transforms[idx] = minority_transform
    else:
        class_transforms[idx] = majority_transform

train_dataset = DFUDataset(train_dir, class_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
]))

# ------------------------------
# Weighted Sampler for Class Imbalance
# ------------------------------
targets = [label for _, label in train_dataset]
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
sample_weights = [class_weights[label] for label in targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# ------------------------------
# Load Pretrained MobileNetV2
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=True)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Loss, Optimizer, LR Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ------------------------------
# Training Parameters
# ------------------------------
num_epochs = 2
best_val_acc = 0
patience = 5
early_stop_counter = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# ------------------------------
# Training and Validation Loops
# ------------------------------
for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct.double() / total
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels)
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / val_total
    val_acc = val_correct.double() / val_total
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    # Confusion Matrix and Report every epoch
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('Confusion Matrix - Epoch' +str(epoch+1)+'.png')

    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_mobilenetv2_dfu.pth')
        best_val_acc = val_acc
        early_stop_counter = 0
        print("Best model saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Test Loss')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuray_n_loss_plot.png')

plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

print(f"\nTraining Complete. Best Val Acc: {best_val_acc:.4f}")
