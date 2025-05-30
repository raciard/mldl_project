import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18
import numpy as np


class PoseModel(nn.Module):
    def __init__(self, num_objects=15, num_keypoints=40):
        super().__init__()
        self.backbone = resnet18(pretrained=True)

       
        self.backbone.fc = nn.Identity()  # Rimuove fully connected finale

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

    def forward(self, x, depth):
        """
        x: tensor [B, 4, H, W], RGB + depth concatenati sui canali
        """
        features = self.backbone(x)  # [B, 512]
        bbox_pred = self.bbox_head(features)  # [B, num_keypoints*2]
        return bbox_pred, ""
