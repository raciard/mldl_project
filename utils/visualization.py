import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import yaml
def load_models_info():
        """Carica le informazioni sui modelli con caching"""
        obj_path = os.path.join("datasets/Linemod_preprocessed", 'models', "models_info.yml")
        with open(obj_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


def show_all(sample, pred_R, pred_T):
    load_models_info()
    visualize_3d_bbox(
        img=sample['original_img'],  # Immagine originale
        pred_R=pred_R,               # Matrice di rotazione predetta
        pred_t=pred_T,               # Vettore traslazione predetto
        gt_R=sample['rotation'].numpy(),  # GT rotation
        gt_t=sample['translation'].numpy(),  # GT translation
        camera_matrix=sample['camera_matrix'].numpy(),
        obj_id=sample['obj_id'].item() + 1,  # Converti in 1-based
        models_info=load_models_info()
    )




def visualize_3d_bbox(img, pred_R=None, pred_t=None, gt_R=None, gt_t=None, 
                     camera_matrix=None, obj_id=None, models_info=None,
                     color_pred=(0, 255, 0), color_gt=(0, 0, 255), thickness=2):
    """
    Visualizza la bounding box 3D sull'immagine
    
    Args:
        img: Immagine RGB (numpy array, torch.Tensor o PIL Image)
        pred_R: Matrice di rotazione predetta (3x3)
        pred_t: Vettore di traslazione predetto (3,)
        gt_R: Matrice di rotazione ground truth (3x3)
        gt_t: Vettore di traslazione ground truth (3,)
        camera_matrix: Matrice della camera (3x3)
        obj_id: ID dell'oggetto (per ottenere le dimensioni del bbox)
        models_info: Dizionario con le info sui modelli
        color_pred: Colore per la bbox predetta (BGR)
        color_gt: Colore per la bbox ground truth (BGR)
        thickness: Spessore delle linee
    """
    # --- Conversione robusta immagine ---
    if isinstance(img, Image.Image):
        img = np.array(img)

    if isinstance(img, torch.Tensor):
        if img.dtype == torch.float32:
            img = (img * 255).byte()
        if img.shape[0] == 3:  # CHW
            img = img.permute(1, 2, 0)
        img = img.numpy()

    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)

    if img.ndim == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    
    # --- Conversione BGR per OpenCV ---
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Ottieni punti 3D del bounding box
    if obj_id is not None and models_info is not None:
        points_3d = get_3d_bbox_points(obj_id, models_info)
    else:
        # Dimensioni di default se non specificate
        points_3d = np.array([
            [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5],
            [ 0.5,  0.5, -0.5],
            [-0.5,  0.5, -0.5],
            [-0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5],
            [ 0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5]
        ], dtype=np.float32)
    
    # Disegna GT se disponibile
    if gt_R is not None and gt_t is not None and camera_matrix is not None:
        gt_points_2d, _ = cv2.projectPoints(points_3d, gt_R, gt_t, camera_matrix, None)
        gt_points_2d = gt_points_2d.reshape(-1, 2).astype(int)
        draw_bbox(img, gt_points_2d, color_gt, thickness, label="GT")
    
    # Disegna predizione se disponibile
    if pred_R is not None and pred_t is not None and camera_matrix is not None:
        pred_points_2d, _ = cv2.projectPoints(points_3d, pred_R, pred_t, camera_matrix, None)
        pred_points_2d = pred_points_2d.reshape(-1, 2).astype(int)
        draw_bbox(img, pred_points_2d, color_pred, thickness, label="Pred")
    
    # Converti in RGB per visualizzazione
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def draw_bbox(img, points_2d, color, thickness, label=None):
    """Disegna la bounding box 3D con connessioni tra i punti"""
    # Base inferiore
    cv2.line(img, tuple(points_2d[0]), tuple(points_2d[1]), color, thickness)
    cv2.line(img, tuple(points_2d[1]), tuple(points_2d[2]), color, thickness)
    cv2.line(img, tuple(points_2d[2]), tuple(points_2d[3]), color, thickness)
    cv2.line(img, tuple(points_2d[3]), tuple(points_2d[0]), color, thickness)
    # Base superiore
    cv2.line(img, tuple(points_2d[4]), tuple(points_2d[5]), color, thickness)
    cv2.line(img, tuple(points_2d[5]), tuple(points_2d[6]), color, thickness)
    cv2.line(img, tuple(points_2d[6]), tuple(points_2d[7]), color, thickness)
    cv2.line(img, tuple(points_2d[7]), tuple(points_2d[4]), color, thickness)
    # Colonne verticali
    cv2.line(img, tuple(points_2d[0]), tuple(points_2d[4]), color, thickness)
    cv2.line(img, tuple(points_2d[1]), tuple(points_2d[5]), color, thickness)
    cv2.line(img, tuple(points_2d[2]), tuple(points_2d[6]), color, thickness)
    cv2.line(img, tuple(points_2d[3]), tuple(points_2d[7]), color, thickness)
    # Etichetta opzionale
    if label:
        cv2.putText(img, label, tuple(points_2d[0]), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, thickness, cv2.LINE_AA)

def get_3d_bbox_points(obj_id, models_info):
    """Ottiene i punti 3D del bounding box per un oggetto specifico"""
    info = models_info[obj_id]
    half_x = info['size_x'] / 2
    half_y = info['size_y'] / 2
    half_z = info['size_z'] / 2
    
    return np.array([
        [-half_x, -half_y, -half_z],
        [ half_x, -half_y, -half_z],
        [ half_x,  half_y, -half_z],
        [-half_x,  half_y, -half_z],
        [-half_x, -half_y,  half_z],
        [ half_x, -half_y,  half_z],
        [ half_x,  half_y,  half_z],
        [-half_x,  half_y,  half_z]
    ], dtype=np.float32)

