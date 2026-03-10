import requests
import os
import io
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse
from app.core.detector import LicensePlateDetector

app = FastAPI(title="LPR High-Speed API")

# Khởi tạo mô hình
detector = LicensePlateDetector(model_path=r"E:\LPR_Project\models\70e\runs\detect\train\weights\best.pt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# --- ENDPOINTS CHÍNH ---

@app.get("/")
async def root():
    return {"message": "mô hình đnga chạy trên fastapi vui lòng truy cập http://127.0.0.1:8000/docs để có thể trải nghiệm"}


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    """Nhận diện ảnh từ máy tính - Phản hồi ngay lập tức"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh!")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Nhận diện và trả về ảnh đã vẽ khung
    _, annotated_image = detector.process_frame(image, draw=True)
    _, img_encoded = cv2.imencode('.jpg', annotated_image)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/predict-url")
async def predict_url(url: str = Query(..., description="Link ảnh trực tiếp (.jpg, .png)")):
    """Nhận diện ảnh qua URL"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)
        response.raise_for_status()
        nparr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, annotated_image = detector.process_frame(image, draw=True)
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi: {str(e)}")