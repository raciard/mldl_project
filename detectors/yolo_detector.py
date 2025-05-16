# detectors/yolo_detector.py
from ultralytics import YOLO  # installa con: pip install ultralytics

class YoloDetector:
    def __init__(self, weights="yolov8n.pt", class_names=None):
        self.model = YOLO(weights)
        self.class_names = class_names if class_names else self.model.names

    def detect(self, image):
        results = self.model.predict(source=image, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })

        return detections

