import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import yaml # added
from PIL import Image # added
import numpy as np # added
# added the following import
from sklearn.model_selection import train_test_split

from model.model import PoseModel
import cv2



class LinemodDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.samples = self.get_all_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}.")

        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed
        )

        self.samples = self.train_samples if split == 'train' else self.test_samples

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Cache per configurazioni e GT
        self.config_cache = {}
        self.models_info_cache = {}
        self.camera_config_cache = {}

        # Carica i dati necessari, ma non direttamente nel costruttore
        self.models_info = self.load_models_info()
        self.gt_cache = {}  # Inizializzo qui la cache GT
        self.points2d_cache = {}

        # Carico i dati GT separatamente
        self.load_all_gt()

    def load_models_info(self):
        """Carica le informazioni sui modelli con caching manuale"""
        if 'models_info' in self.models_info_cache:
            return self.models_info_cache['models_info']
        
        obj_path = os.path.join(self.dataset_root, 'models', "models_info.yml")
        with open(obj_path, 'r') as f:
            models_info = yaml.load(f, Loader=yaml.FullLoader)
        
        # Memorizza nella cache
        self.models_info_cache['models_info'] = models_info
        return models_info
    
    def get_all_samples(self):
        samples = []
        for folder_id in range(1, 16):
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb")
            if os.path.exists(folder_path):
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])
        return samples

    def load_gt_for_folder(self, folder_id):
        """Carica i GT per una cartella specifica con caching manuale"""
        if folder_id in self.gt_cache:
            return self.gt_cache[folder_id]
        
        pose_file = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "gt.yml")
        if os.path.exists(pose_file):
            with open(pose_file, 'r') as f:
                gt_data = yaml.load(f, Loader=yaml.FullLoader)
        else:
            gt_data = {}

        # Memorizza nella cache
        self.gt_cache[folder_id] = gt_data
        return gt_data

    def load_all_gt(self):
        """Precarica tutti i dati GT con caching"""
        for folder_id in range(1, 16):
            self.load_gt_for_folder(folder_id)

    def load_camera_config(self, folder_id):
        """Carica la configurazione della camera con caching manuale"""
        if folder_id in self.camera_config_cache:
            return self.camera_config_cache[folder_id]
        
        cam_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", 'info.yml')
        with open(cam_path, 'r') as f:
            camera_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Memorizza nella cache
        self.camera_config_cache[folder_id] = camera_config
        return camera_config

    def load_config(self, folder_id):
        """Ottiene la configurazione camera e oggetti con caching manuale"""
        if folder_id not in self.config_cache:
            cam = self.load_camera_config(folder_id)
            obj = self.models_info
            self.config_cache[folder_id] = (cam, obj)
        return self.config_cache[folder_id]

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def load_normal_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return transforms.ToTensor()(img)

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.gt_cache.get(folder_id, {})
        str_id = sample_id
        
        if str_id not in pose_data:
            print(f"id: {str_id},{pose_data}")
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        pose = pose_data[str_id][0]
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        bbox = np.array(pose['obj_bb'], dtype=np.float32)
        obj_id = pose['obj_id']

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return translation, rotation, bbox, obj_id

    def get_3d_bbox_projection(self, obj_id, rotation, translation, camera_matrix):
        """Proietta i punti 3D del bounding box nell'immagine 2D"""
        points_3d = PoseModel().get_3d_bbox_points(obj_id, self.models_info)
        points_2d, _ = cv2.projectPoints(points_3d, rotation, translation, 
                                        camera_matrix, None)
        return points_2d.squeeze()

    def normalize_points(self, points_2d, bbox):
        """Normalizza i punti 2D rispetto alla bounding box"""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalizza tra 0 e 1 rispetto alla bounding box
        points_norm = (points_2d - np.array([x_min, y_min])) / np.array([width, height])
        return points_norm.astype(np.float32).flatten()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder_id, sample_id = self.samples[idx]
        camera_intrinsics, _ = self.load_config(folder_id)
        camera_matrix = np.array(camera_intrinsics[0]['cam_K']).reshape(3, 3)

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        img = self.load_image(img_path)
        
        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)
        
        # Proietta i punti 3D del bounding box
        if f"{folder_id}-{sample_id}" not in self.points2d_cache:
            points_2d = self.get_3d_bbox_projection(obj_id, rotation, translation, camera_matrix)
        
            points_norm = self.normalize_points(points_2d, bbox) 
            self.points2d_cache[f'{folder_id}-{sample_id}'] = points_norm
        else:
            points_norm = self.points2d_cache[f'{folder_id}-{sample_id}'] 
        
        # Crop dell'immagine usando la bounding box con padding
        img_pil = transforms.ToPILImage()(img)
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Aggiungi padding del 20%
        pad_x = int(0.2 * (x_max - x_min))
        pad_y = int(0.2 * (y_max - y_min))
        
        x_min = max(0, x_min - pad_x)
        y_min = max(0, y_min - pad_y)
        x_max = min(img_pil.width, x_max + pad_x)
        y_max = min(img_pil.height, y_max + pad_y)
        
        cropped = img_pil.crop((x_min, y_min, x_max, y_max))
        cropped_resized = transforms.Resize((224, 224))(cropped)
        cropped_tensor = self.transform(cropped_resized)
        
        return {
            "rgb": cropped_tensor,
            "points_2d": torch.tensor(points_norm),
            "obj_id": torch.tensor(obj_id - 1),  # Converti in indice 0-based
            "original_img": self.load_normal_image(img_path),
            "bbox": torch.tensor(bbox),
            "rotation": torch.tensor(rotation),
            "translation": torch.tensor(translation),
            "camera_matrix": torch.tensor(camera_matrix),
        }

