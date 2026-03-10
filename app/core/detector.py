from ultralytics import YOLO
import cv2
import numpy as np

class LicensePlateDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def process_frame(self, frame, draw=False):

        if frame is None:
            return [], None

        # resize ảnh để tránh timeout trên Render
        frame = cv2.resize(frame, (640, 640))

        # chạy model
        results = self.model(frame, imgsz=640, conf=0.25, verbose=False)

        detections = []

        annotated_frame = results[0].plot() if draw else frame

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls]

                detections.append({
                    "bbox": [round(x, 2) for x in [x1, y1, x2, y2]],
                    "confidence": round(conf, 2),
                    "class_name": label
                })

        return detections, annotated_frame