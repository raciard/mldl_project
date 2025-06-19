import numpy as np
import trimesh  # for loading 3D model files
import os
from scipy.spatial import cKDTree


def compute_add(model_path, R_gt, t_gt, R_pred, t_pred):
    """
    Compute the ADD (Average Distance of Model Points) metric.

    Parameters:
        model_path (str): Path to 3D model file (e.g. .ply or .obj).
        R_gt (np.ndarray): Ground truth rotation matrix (3x3).
        t_gt (np.ndarray): Ground truth translation vector (3,).
        R_pred (np.ndarray): Predicted rotation matrix (3x3).
        t_pred (np.ndarray): Predicted translation vector (3,).

    Returns:
        float: The ADD metric (average point-wise distance).
    """
    # Load the model and extract vertices
    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    model_points = mesh.vertices  # (N, 3)

    # Ensure translations are (3,)
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()

    pts_gt = (R_gt @ model_points.T).T + t_gt
    pts_pred = (R_pred @ model_points.T).T + t_pred

    distances = np.linalg.norm(pts_gt - pts_pred, axis=1)
    return np.mean(distances)


def compute_adds(model_path, R_gt, t_gt, R_pred, t_pred):
    """
    Compute the ADD-S (Average Distance of Model Points - Symmetric) metric.

    Parameters:
        model_path (str): Path to 3D model file (e.g. .ply or .obj).
        R_gt (np.ndarray): Ground truth rotation matrix (3x3).
        t_gt (np.ndarray): Ground truth translation vector (3,).
        R_pred (np.ndarray): Predicted rotation matrix (3x3).
        t_pred (np.ndarray): Predicted translation vector (3,).

    Returns:
        float: The ADD-S metric (average closest point distance).
    """
    # Load the model and extract vertices
    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    model_points = mesh.vertices  # (N, 3)

    # Ensure translations are (3,)
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()

    pts_gt = (R_gt @ model_points.T).T + t_gt
    pts_pred = (R_pred @ model_points.T).T + t_pred

    # Use nearest neighbor distance (for symmetric objects)
    tree = cKDTree(pts_pred)
    distances, _ = tree.query(pts_gt, k=1)

    return np.mean(distances)


def compute_adds_percent(model_path, R_gt, t_gt, R_pred, t_pred):
    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    model_points = mesh.vertices  # (N, 3)

    # Ensure translations are (3,)
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()

    pts_gt = (R_gt @ model_points.T).T + t_gt
    pts_pred = (R_pred @ model_points.T).T + t_pred

    # Use nearest neighbor distance (for symmetric objects)
    tree = cKDTree(pts_pred)
    distances, _ = tree.query(pts_gt, k=1)
    add = np.mean(distances)

    # Compute model diameter
    diameter = np.linalg.norm(
        np.max(model_points, axis=0) - np.min(model_points, axis=0)
    )

    add_percent = (1 - (add / diameter)) * 100
    add_percent = max(0.0, min(100.0, add_percent))  # Clip to [0, 100]

    return add, add_percent, diameter


def compute_add_percent(model_path, R_gt, t_gt, R_pred, t_pred):
    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    model_points = mesh.vertices  # (N, 3)

    # Ensure translations are (3,)
    t_gt = np.asarray(t_gt).flatten()
    t_pred = np.asarray(t_pred).flatten()

    # Ground truth and predicted transformations
    pts_gt = (R_gt @ model_points.T).T + t_gt
    pts_pred = (R_pred @ model_points.T).T + t_pred

    # Compute ADD
    distances = np.linalg.norm(pts_gt - pts_pred, axis=1)
    add = np.mean(distances)

    # Compute model diameter
    diameter = np.linalg.norm(
        np.max(model_points, axis=0) - np.min(model_points, axis=0)
    )

    add_percent = (1 - (add / diameter)) * 100
    add_percent = max(0.0, min(100.0, add_percent))  # Clip to [0, 100]

    return add, add_percent, diameter
