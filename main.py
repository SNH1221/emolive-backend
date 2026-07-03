from fastapi import FastAPI, UploadFile, File, Form
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "michellejieli/emotion_text_classifier"
FACE_MODEL = "dima806/facial_emotions_image_detection"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

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
        json={
            "inputs": text,
            "parameters": {"top_k": 7}
        }
    )

    raw = response.json()

    if isinstance(raw, list) and len(raw) > 0:
        emotions_raw = raw[0] if isinstance(raw[0], list) else raw
        returned_labels = {e["label"].lower(): e["score"] for e in emotions_raw}

        all_scores = []
        for emotion in ALL_EMOTIONS:
            all_scores.append({
                "label": emotion,
                "score": returned_labels.get(emotion, 0.0)
            })

        all_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [all_scores]
        }

    return {
        "text": text,
        "emotions": raw
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
