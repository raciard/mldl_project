from ultralytics import YOLO
import os
def train_yolo(data_root):
    model = YOLO('yolo11n.pt')  # puoi scegliere anche yolov8s.pt
    model.train(
        data=os.path.join(data_root, 'data.yaml'),
        epochs=10,
        imgsz=640,
        batch=16,
        name='yolo-linemod'
    )
