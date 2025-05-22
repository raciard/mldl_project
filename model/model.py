import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18
import numpy as np


class PoseModel(nn.Module):
    def __init__(self, num_objects=15, num_keypoints=20):
        super().__init__()
        # Usiamo ResNet18 come backbone invece di VGG per maggiore efficienza
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Rimuoviamo il fully connected finale

        # Testa per la predizione dei punti 2D (8 corners * 2 coordinate)
        self.bbox_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox_pred = self.bbox_head(features)
        return bbox_pred, ""
