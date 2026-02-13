from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
from transformers import pipeline
import cv2
import numpy as np

app = FastAPI()

# Text Emotion Model
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

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

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):
    result = emotion_pipeline(text)

    return {
        "text": text,
        "emotions": result[0]
    }
