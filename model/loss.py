import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn

class PoseLoss(nn.Module):
    """
    Modulo PyTorch per la loss combinata nella 6D pose estimation.
    Combina MSE per traslazione e distanza sui quaternioni (o Geodesic Loss) per rotazione.
    """
    def __init__(self, w_t=100.0, w_r=1.0, rotation_loss_type="quaternion"):
        """
        Inizializza il modulo di loss.
        
        Args:
            w_t (float): Peso per la loss di traslazione
            w_r (float): Peso per la loss di rotazione
            rotation_loss_type (str): Tipo di loss per rotazione ("quaternion" o "geodesic")
        """
        super(PoseLoss, self).__init__()
        self.w_t = w_t
        self.w_r = w_r
        self.rotation_loss_type = rotation_loss_type
        if rotation_loss_type not in ["quaternion", "geodesic"]:
            raise ValueError("rotation_loss_type deve essere 'quaternion' o 'geodesic'")
        
        # MSE per traslazione
       # self.mse_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, pred_trans, pred_quat, gt_trans, gt_quat):
        """
        Calcola la loss combinata.
        
        Args:
            pred_trans (torch.Tensor): [batch, 3], traslazioni predette (normalizzate)
            pred_quat (torch.Tensor): [batch, 4], quaternioni predetti (normalizzati)
            gt_trans (torch.Tensor): [batch, 3], traslazioni ground truth (normalizzate)
            gt_quat (torch.Tensor): [batch, 4], quaternioni ground truth (normalizzati)
        
        Returns:
            torch.Tensor: Loss totale
        """
        # Loss traslazione (MSE)
        #trans_loss = self.mse_loss(pred_trans, gt_trans)
        
        # Loss rotazione
        if self.rotation_loss_type == "quaternion":
            # Distanza sui quaternioni: 1 - |q · q̂|
            dot_product = torch.abs(torch.sum(pred_quat * gt_quat, dim=-1))
            rot_loss = torch.mean(1 - dot_product)
        
        # Loss totale
        return rot_loss
