# scripts/generate_yolo_annotations.py
import os
import json
import cv2
from tqdm import tqdm
from detectors.yolo_detector import YoloDetector  # da te definito
from data.dataset import LinemodDataset



def generate_yolo_annotations(weights, dataset):
    OUTPUT_JSON = "datasets/yolo_annotations.json"

# Inizializza YOLO
    yolo = YoloDetector(weights=weights)  # personalizza

# Trova immagini

    annotations = {}

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        folder_id, sample_id = dataset.samples[i]
        img_rgb = sample['original_img'].unsqueeze(0)

        detections = yolo.detect(img_rgb)

        annotations[f"{folder_id}-{sample_id}"] = detections  # già in formato [{class_id, bbox, confidence, class_name}]

# Salva su file
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(annotations, f, indent=2)

