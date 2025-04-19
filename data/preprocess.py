import torch
import numpy as np

def normalize_translation(translations, scales=None):
    """
    Normalizza le traslazioni per LINEMOD.
    
    Args:
        translations: [batch, 3], vettore [x, y, z] in mm
        scales: dict con 'x': [min, max], 'y': [min, max], 'z': [min, max]
    
    Returns:
        translations_norm: [batch, 3], traslazioni normalizzate
    """
    if scales is None:
        scales = {
            'x': [-300, 300],  # mm
            'y': [-200, 200],  # mm
            'z': [400, 1200]   # mm
        }
    
    translations = translations.clone() if isinstance(translations, torch.Tensor) else np.copy(translations)
    translations_norm = translations.copy()
    
    translations_norm[:, 0] = translations[:, 0] / 300  # x
    translations_norm[:, 1] = translations[:, 1] / 200  # y
    translations_norm[:, 2] = (translations[:, 2] - 400) / 800  # z
    
    return translations_norm

def denormalize_translation(translations_norm, scales=None):
    """
    Denormalizza le traslazioni per LINEMOD.
    
    Args:
        translations_norm: [batch, 3], traslazioni normalizzate
        scales: dict con 'x': [min, max], 'y': [min, max], 'z': [min, max]
    
    Returns:
        translations: [batch, 3], traslazioni in mm
    """
    if scales is None:
        scales = {
            'x': [-300, 300],
            'y': [-200, 200],
            'z': [400, 1200]
        }
    
    translations = translations_norm.clone() if isinstance(translations_norm, torch.Tensor) else np.copy(translations_norm)
    
    translations[:, 0] = translations_norm[:, 0] * 300
    translations[:, 1] = translations_norm[:, 1] * 200
    translations[:, 2] = translations_norm[:, 2] * 800 + 400
    
    return translations
