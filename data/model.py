import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18

class PoseModel(nn.Module):

    def __init__(self, num_objects=15):
        super().__init__()
        # Usiamo ResNet18 come backbone invece di VGG per maggiore efficienza
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Rimuoviamo il fully connected finale
        
        # Testa per la predizione dei punti 2D (8 corners * 2 coordinate)
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16)# 8 punti * 2 coordinate
        )
        
        # Testa per il classificatore di simmetria (4 range per oggetti approssimativamente simmetrici)
        self.symmetry_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_objects * 4),
            nn.Sigmoid()
        )
        
        # Cache per i modelli 3D
        self.model_points_cache = {}

    def forward(self, x):
        features = self.backbone(x)
        bbox_pred = self.bbox_head(features)
        symmetry_pred = self.symmetry_head(features)
        return bbox_pred, symmetry_pred

    def get_3d_bbox_points(self, obj_id, models_info):
        """Ottiene i punti 3D del bounding box per un oggetto"""
        if obj_id not in self.model_points_cache:
            info = models_info[obj_id]
            x_size = info['size_x']
            y_size = info['size_y']
            z_size = info['size_z']
            
            half_x = x_size / 2
            half_y = y_size / 2
            half_z = z_size / 2
            
            points_3d = np.array([
                [-half_x, -half_y, -half_z],
                [half_x, -half_y, -half_z],
                [half_x, half_y, -half_z],
                [-half_x, half_y, -half_z],
                [-half_x, -half_y, half_z],
                [half_x, -half_y, half_z],
                [half_x, half_y, half_z],
                [-half_x, half_y, half_z]
            ], dtype=np.float32)
            
            self.model_points_cache[obj_id] = points_3d
        
        return self.model_points_cache[obj_id]
