import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import multiprocessing
import os
from tqdm import tqdm  # Barra di progresso
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os


from model.model import PoseModel
from data.dataset import LinemodDataset


def train_model(batch_size = 128, num_epochs=10):
    # Configurazione
    dataset_root = "datasets/Linemod_preprocessed/"
    learning_rate = 0.001
    checkpoint_dir = "checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Dataset e DataLoader
    train_dataset = LinemodDataset(dataset_root, split='train')
    val_dataset = LinemodDataset(dataset_root, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=4)
    
    # Modello e ottimizzatore
    model = PoseModel(num_objects=15).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images = batch['rgb'].cuda()
            targets = batch['points_2d'].cuda()
            
            pred_points, _ = model(images)
            batch_size = images.size(0)
            selected_preds = torch.zeros(batch_size, 16).cuda()
            for i in range(batch_size):
                selected_preds[i] = pred_points[i]
            
            loss = criterion(selected_preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
                images = batch['rgb'].cuda()
                targets = batch['points_2d'].cuda()
                
                pred_points, _ = model(images)
                batch_size = images.size(0)
                selected_preds = torch.zeros(batch_size, 16).cuda()
                for i in range(batch_size):
                    selected_preds[i] = pred_points[i]
                
                loss = criterion(selected_preds, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # 🔄 Salvataggio "last"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss
        }, os.path.join(checkpoint_dir, "last.pth"))
        
        # ⭐ Salvataggio "best"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss
            }, os.path.join(checkpoint_dir, "best.pth"))
            print("✅ Nuovo modello migliore salvato.")
    
    return model
