"""
Script to train the Target (Victim) CNN: Multi-view DenseNet121.
This model will be used as the judge for our GANs and for the custom adversarial loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
from tqdm import tqdm

from src.data.dataset import NoteShieldDataset
from src.models.target_model import TargetCNN

def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"  # Path to dataset root
    batch_size = 16
    epochs = 20
    learning_rate = 1e-4
    img_size = (224, 224) # Standard for DenseNet121
    save_path = "checkpoints/target_model_best.pth"

    print(f"Using device: {device}")

    # 2. Data Preparation
    # DenseNet121 expects specific ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(img_size),
        # Note: dataset.py already converts to tensor [0, 1] via TF.to_tensor
        # We just need the ImageNet normalization here
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = NoteShieldDataset(data_dir=data_dir, transform=transform)
    
    if len(full_dataset) == 0:
        print(f"Error: No data found in {data_dir}. Ensure 'real_notes' and 'fake_notes' folders exist.")
        return

    # Split into Train (80%) and Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # 3. Model, Loss, Optimizer
    model = TargetCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] (Train)", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / len(val_dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved with Val Acc: {val_acc:.4f}")

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
