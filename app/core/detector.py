from ultralytics import YOLO
import cv2
import numpy as np

class LicensePlateDetector:
    def __init__(self, model_path: str):
        # Khởi tạo mô hình. Nên dùng .to('cuda') nếu máy có GPU, không thì mặc định là CPU
        self.model = YOLO(model_path)

    def process_frame(self, frame, draw=False):
        """
        Xử lý một khung hình đơn lẻ.
        Đầu vào: frame (numpy array từ cv2 hoặc PIL)
        Đầu ra: (danh sách kết quả, khung hình đã vẽ)
        """
        # Kiểm tra nếu frame rỗng (tránh lỗi khi link video bị đứt)
        if frame is None:
            return [], None

        # Chạy inference
        results = self.model(frame, verbose=False) # verbose=False để console sạch hơn
        detections = []
        
        # result.plot() vẽ khung và nhãn lên ảnh
        annotated_frame = results[0].plot() if draw else frame
        
        for result in results:
            for box in result.boxes:
                # Trích xuất dữ liệu
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = result.names[cls] # Lấy tên class (ví dụ: 'license_plate')

                detections.append({
                    "bbox": [round(x, 2) for x in [x1, y1, x2, y2]],
                    "confidence": round(conf, 2),
                    "class_name": label
                })
        
        return detections, annotated_frame