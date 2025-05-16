import os
import shutil
import yaml
from tqdm import tqdm
from PIL import Image

from data.dataset import LinemodDataset  # importa la tua classe
import torch

def prepare_yolo_dataset(train_dataset, val_dataset, dataset_root, output_root):
    os.makedirs(output_root, exist_ok=True)

    # Crea cartelle YOLO
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_root, f'images/{split}'), exist_ok=True)
        os.makedirs(os.path.join(output_root, f'labels/{split}'), exist_ok=True)

    # Prepara dataset
    for dataset, split in zip([train_dataset, val_dataset],['train', 'val']):

        print(f"📂 Processing split: {split} with {len(dataset)} samples...")
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            folder_id, sample_id = dataset.samples[i]

            # Prepara path
            image_name = f"{folder_id:02d}_{sample_id:04d}.png"
            image_src_path = os.path.join(dataset_root, 'data', f"{folder_id:02d}", 'rgb', f"{sample_id:04d}.png")
            image_dst_path = os.path.join(output_root, f'images/{split}', image_name)
            label_dst_path = os.path.join(output_root, f'labels/{split}', image_name.replace('.png', '.txt'))

            # Copia immagine
            shutil.copy(image_src_path, image_dst_path)

            # YOLO bbox [x_center, y_center, width, height] normalizzata
            bbox = sample['bbox'].numpy()
            img = Image.open(image_src_path)
            img_w, img_h = img.size

            x_min, y_min, x_max, y_max = bbox
            xc = (x_min + x_max) / 2 / img_w
            yc = (y_min + y_max) / 2 / img_h
            w = (x_max - x_min) / img_w
            h = (y_max - y_min) / img_h

            class_id = sample['obj_id'].item()

            with open(label_dst_path, 'w') as f:
                f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Scrive file data.yaml
    data_yaml = {
        'train': os.path.abspath(os.path.join(output_root, 'images/train')),
        'val': os.path.abspath(os.path.join(output_root, 'images/val')),
        'nc': 15,
        'names': [str(i) for i in range(15)]
    }
    with open(os.path.join(output_root, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"✅ YOLO dataset pronto in: {output_root}")
