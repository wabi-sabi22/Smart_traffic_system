import io
import cv2
import numpy as np
import requests

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse

from huggingface_hub import hf_hub_download

from app.core.detector import LicensePlateDetector


app = FastAPI(title="LPR High-Speed API")

# =============================
# LOAD MODEL FROM HUGGINGFACE
# =============================

model_path = hf_hub_download(
    repo_id="nhu445/lpr-yolo11n",
    filename="best.pt"
)

detector = LicensePlateDetector(model_path=model_path)


HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# =============================
# ROOT ENDPOINT
# =============================

@app.get("/")
async def root():
    return {
        "message": "License Plate Recognition API is running",
        "docs": "/docs"
    }


# =============================
# IMAGE UPLOAD DETECTION
# =============================

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported"
        )

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detections, annotated_image = detector.process_frame(image, draw=True)

    _, img_encoded = cv2.imencode(".jpg", annotated_image)

    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )


# =============================
# URL IMAGE DETECTION
# =============================

@app.post("/predict-url")
async def predict_url(
    url: str = Query(..., description="Direct image URL (.jpg, .png)")
):

    try:

        response = requests.get(url, headers=HEADERS, timeout=10)

        response.raise_for_status()

        nparr = np.frombuffer(response.content, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        detections, annotated_image = detector.process_frame(image, draw=True)

        _, img_encoded = cv2.imencode(".jpg", annotated_image)

        return StreamingResponse(
            io.BytesIO(img_encoded.tobytes()),
            media_type="image/jpeg"
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )