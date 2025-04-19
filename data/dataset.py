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


from .preprocess import normalize_translation


class LinemodDataset(Dataset):
    def __init__(self, dataset_root, split='train', train_ratio=0.8, seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.scales = {
            'x': [-300, 300],
            'y': [-200, 200],
            'z': [400, 1200]
        }
        self.samples = self.get_all_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_root}.")

        self.train_samples, self.test_samples = train_test_split(
            self.samples, train_size=self.train_ratio, random_state=self.seed
        )

        self.samples = self.train_samples if split == 'train' else self.test_samples

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(           # Normalizza con media e std di ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

        # Cache config and GT
        self.config_cache = {}
        self.gt_cache = self.load_all_gt()

    def get_all_samples(self):
        samples = []
        for folder_id in range(1, 16):
            folder_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "rgb")
            if os.path.exists(folder_path):
                sample_ids = sorted([int(f.split('.')[0]) for f in os.listdir(folder_path) if f.endswith('.png')])
                samples.extend([(folder_id, sid) for sid in sample_ids])
        return samples

    def load_all_gt(self):
        """Preload all GT data into memory to avoid reading YAML multiple times."""
        gt_data = {}
        for folder_id in range(1, 16):
            pose_file = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", "gt.yml")
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    pose_data = yaml.load(f, Loader=yaml.FullLoader)
                gt_data[folder_id] = pose_data
        return gt_data

    def load_config(self, folder_id):
        if folder_id not in self.config_cache:
            cam_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", 'info.yml')
            obj_path = os.path.join(self.dataset_root, 'models', f"models_info.yml")

            with open(cam_path, 'r') as f:
                cam = yaml.load(f, Loader=yaml.FullLoader)
            with open(obj_path, 'r') as f:
                obj = yaml.load(f, Loader=yaml.FullLoader)

            self.config_cache[folder_id] = (cam, obj)
        return self.config_cache[folder_id]

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

    def load_6d_pose(self, folder_id, sample_id):
        pose_data = self.gt_cache.get(folder_id, {})
        if sample_id not in pose_data:
            raise KeyError(f"Sample ID {sample_id} not found in gt.yml for folder {folder_id}.")

        pose = pose_data[sample_id][0]
        translation = np.array(pose['cam_t_m2c'], dtype=np.float32)
        rotation = np.array(pose['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
        bbox = np.array(pose['obj_bb'], dtype=np.float32)
        obj_id = np.array(pose['obj_id'], dtype=np.float32)

        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        bbox = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        return translation, rotation, bbox, obj_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a dataset sample."""
        folder_id, sample_id = self.samples[idx]

        # Load the correct camera intrinsics and object info for this folder
        camera_intrinsics, objects_info = self.load_config(folder_id)

        img_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"rgb/{sample_id:04d}.png")
        depth_path = os.path.join(self.dataset_root, 'data', f"{folder_id:02d}", f"depth/{sample_id:04d}.png")

        img = self.load_image(img_path)
        #depth = self.load_depth(depth_path)
        #point_cloud = self.load_point_cloud(depth.numpy(), camera_intrinsics)
        #point_cloud = torch.tensor(np.asarray(point_cloud.points), dtype=torch.float32)
        translation, rotation, bbox, obj_id = self.load_6d_pose(folder_id, sample_id)

        # Crop image using bbox
        img_pil = transforms.ToPILImage()(img)
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped = img_pil.crop((x_min, y_min, x_max, y_max))
        cropped_resized = transforms.Resize((224, 224))(cropped)
        cropped_tensor = self.transform(cropped_resized)

        return {
            "rgb": img,
            #"depth": torch.tensor(depth, dtype=torch.float32),
            #"point_cloud": point_cloud,
            "camera_intrinsics": camera_intrinsics[0]['cam_K'],
            #"objects_info": objects_info,
            "translation": torch.tensor(normalize_translation(translation)),
            "rotation": torch.tensor(rotation),
            "bbox": torch.tensor(bbox),
            "obj_id": torch.tensor(obj_id - 1),
            "cropped_img": cropped_tensor
        }
