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
   
    
    return translations

def denormalize_translation(translations_norm, scales=None):
    
    
    return translations_norm
