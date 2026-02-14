from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
import cv2
import numpy as np
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

@app.get("/")
def home():
    return {"message": "EmoLive Backend Running 🚀"}


# =========================
# FACE EMOTION DETECTION
# =========================
@app.post("/detect-face-emotion")
async def detect_face_emotion(file: UploadFile = File(...)):

    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = DeepFace.analyze(
        img,
        actions=['emotion'],
        enforce_detection=False
    )

    emotion_data = result[0]

    return {
        "dominant_emotion": emotion_data["dominant_emotion"],
        "emotion_scores": emotion_data["emotion"]
    }


# =========================
# TEXT EMOTION DETECTION
# =========================
@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    API_URL = "https://router.huggingface.co/hf-inference/models/j-hartmann/emotion-english-distilroberta-base"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": text
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    return {
        "text": text,
        "emotions": response.json()
    }
