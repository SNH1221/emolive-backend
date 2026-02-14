from fastapi import FastAPI, UploadFile, File, Form
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
FACE_MODEL = "dima806/facial_emotions_image_detection"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.get("/")
def root():
    return {"message": "EmoLive Backend Running"}

# ---------------- TEXT EMOTION ----------------

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):
    url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"

    response = requests.post(
        url,
        headers=HEADERS,
        json={"inputs": text}
    )

    return {
        "text": text,
        "emotions": response.json()
    }

# ---------------- FACE EMOTION ----------------

@app.post("/detect-face-emotion")
async def detect_face_emotion(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    url = f"https://router.huggingface.co/hf-inference/models/{FACE_MODEL}"

    response = requests.post(
        url,
        headers=HEADERS,
        json={"inputs": base64_image}
    )

    return response.json()
