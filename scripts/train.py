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
from scipy.spatial.transform import Rotation
import os


from ..model.model import PoseModel
from ..model.loss import PoseLoss
from ..data.dataset import LinemodDataset


# Funzione per convertire matrice di rotazione in quaternione
def rotation_matrix_to_quaternion(rotation):
    """
    Converte una matrice di rotazione 3x3 in quaternione (w, x, y, z).
    Input: rotation (torch.Tensor, shape: [3, 3] o [batch, 3, 3])
    Output: quaternione (torch.Tensor, shape: [4] o [batch, 4])
    """
    if rotation.dim() == 2:
        rotation = rotation.unsqueeze(0)  # Aggiungi dimensione batch
    r = Rotation.from_matrix(rotation.cpu().numpy())
    quaternion = r.as_quat()  # Formato: [x, y, z, w]
    quaternion = torch.tensor(quaternion, dtype=torch.float32)  # Converti in tensore
    # Riordina in [w, x, y, z]
    quaternion = quaternion[:, [3, 0, 1, 2]]
    return quaternion.squeeze() if rotation.shape[0] == 1 else quaternion



# Funzione di addestramento con salvataggio ad ogni epoca
def train_model(model, train_loader, val_loader=None, num_epochs=20, device='cuda', save_dir='checkpoints'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Crea la directory per i checkpoint se non esiste
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')  # Per salvare il modello migliore (opzionale)
    pose_loss = PoseLoss()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch['cropped_img'].to(device)
            obj_ids = batch['obj_id'].long().to(device)
            gt_trans = batch['translation'].to(device)
            gt_rot = batch['rotation'].to(device)

            # Converti matrice di rotazione in quaternione
            gt_quat = torch.stack([rotation_matrix_to_quaternion(r) for r in gt_rot]).to(device)

            # Forward pass
            pred_trans, pred_rot = model(images, obj_ids)

            # Calcola la loss
            loss = pose_loss.foward(pred_trans, pred_rot, gt_trans, gt_quat)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoca [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validazione (opzionale)
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['cropped_img'].to(device)
                    obj_ids = batch['obj_id'].long().to(device)
                    gt_trans = batch['translation'].to(device)
                    gt_rot = batch['rotation'].to(device)
                    gt_quat = torch.stack([rotation_matrix_to_quaternion(r) for r in gt_rot]).to(device)

                    pred_trans, pred_rot = model(images, obj_ids)
                    val_loss += pose_loss(pred_trans, pred_rot, gt_trans, gt_quat).item()
            avg_val_loss = val_loss / len(val_loader)
            print(f'Validazione, Loss: {avg_val_loss:.4f}')

            # Aggiorna lo scheduler
            scheduler.step(avg_val_loss)

            # Salva il modello migliore (opzionale)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'pose_cnn_best.pth'))
                print(f'Salvato modello migliore con loss di validazione: {best_val_loss:.4f}')

        # Salva il modello per l'epoca corrente
        checkpoint_path = os.path.join(save_dir, f'pose_cnn_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Salvato modello: {checkpoint_path}')
