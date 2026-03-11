import gradio as gr
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from app.core.detector import LicensePlateDetector

# =============================
# LOAD MODEL 
# =============================
model_path = hf_hub_download(
    repo_id="nhu445/lpr-yolo11n", 
    filename="best.pt"
)
detector = LicensePlateDetector(model_path=model_path)

# =============================
# LOGIC XỬ LÝ CHO GRADIO
# =============================
def predict(input_type, image=None, video=None):
    if input_type == "Image" and image is not None:
        # Xử lý ảnh
        detections, annotated_image = detector.process_frame(image, draw=True)
        # Chuyển BGR sang RGB để Gradio hiển thị đúng
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), None
    
    elif input_type == "Video" and video is not None:
        # Xử lý video (Logic xử lý từng frame)
        cap = cv2.VideoCapture(video)
        output_path = "output_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Lấy thông số video (Detector của bạn resize về 640x640)
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 640))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            _, annotated_frame = detector.process_frame(frame, draw=True)
            out.write(annotated_frame)
            
        cap.release()
        out.release()
        return None, output_path
    
    return None, None

# =============================
# GIAO DIỆN GRADIO 
# =============================
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# 🚗 Smart LPR System - Vietnam License Plate Recognition")
    gr.Markdown("Dự án nhận diện biển số xe sử dụng YOLOv11 & OCR. Hệ thống tự động phát hiện và tối ưu hình ảnh.")

    with gr.Tab("Dành cho Ảnh"):
        img_input = gr.Image(label="Tải ảnh biển số xe")
        img_output = gr.Image(label="Kết quả nhận diện")
        btn_img = gr.Button("Phân tích ảnh")
        btn_img.click(fn=lambda x: predict("Image", image=x), inputs=img_input, outputs=[img_output, gr.State()])

    with gr.Tab("Dành cho Video"):
        vid_input = gr.Video(label="Tải video bãi xe/đường phố")
        vid_output = gr.Video(label="Video đã xử lý")
        btn_vid = gr.Button("Bắt đầu xử lý")
        btn_vid.click(fn=lambda x: predict("Video", video=x), inputs=vid_input, outputs=[gr.State(), vid_output])

demo.launch()