import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Definizione del modello PoseCNN
class PoseModel(nn.Module):
    def __init__(self, num_objects=15):
        super(PoseCNN, self).__init__()

        # Backbone: ResNet-18 pre-addestrata
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Rimuove FC layer

        # Embedding per obj_id (opzionale)
        self.num_objects = num_objects
        self.obj_embedding = nn.Embedding(num_objects, 32)  # 32 dimensioni per oggetto

        # Branca di traslazione: prevede (x, y, z)
        self.translation_head = nn.Sequential(
            nn.Linear(512 + 32, 256),  # 512 (ResNet) + 32 (embedding)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)  # Output: (x, y, z)
        )

        # Branca di rotazione: prevede quaternione (4 valori)
        self.rotation_head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
            nn.Tanh()  # Normalizza tra [-1, 1]
        )

    def forward(self, x, obj_id):
        # Input: x (immagine, [batch, 3, 224, 224]), obj_id ([batch])
        features = self.backbone(x)  # [batch, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 512]

        # Embedding per obj_id
        obj_embed = self.obj_embedding(obj_id)  # [batch, 32]
        features = torch.cat([features, obj_embed], dim=1)  # [batch, 512 + 32]

        # Previsione traslazione
        translation = self.translation_head(features)  # [batch, 3]

        # Previsione rotazione
        rotation = self.rotation_head(features)  # [batch, 4]
        rotation = F.normalize(rotation, p=2, dim=1)  # Normalizza il quaternione

        return translation, rotation

