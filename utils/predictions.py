import cv2
import torch
import numpy as np

def estimate_pose(model, dataset, idx):
    """Stima la posa 3D per un'immagine del dataset"""
    sample = dataset[idx]
    img = sample['rgb'].unsqueeze(0).cuda()
    
    with torch.no_grad():
        pred_points_norm, _ = model(img)
    
    pred_points_norm = pred_points_norm.cpu().numpy().reshape(8, 2)
    
    # Converti i punti normalizzati in coordinate immagine
    bbox = sample['bbox'].numpy()
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    pred_points_img = np.zeros_like(pred_points_norm)
    pred_points_img[:, 0] = pred_points_norm[:, 0] * width + x_min
    pred_points_img[:, 1] = pred_points_norm[:, 1] * height + y_min
    
    # Risolvi PnP
    obj_id = sample['obj_id'].item() + 1  # Converti in ID 1-based
    points_3d = model.get_3d_bbox_points(obj_id, dataset.models_info)
    camera_matrix = sample['camera_matrix'].numpy()
    
    _, rvec, tvec = cv2.solvePnP(points_3d, pred_points_img, camera_matrix, None)
    
    # Converti rvec in matrice di rotazione
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec, pred_points_img
