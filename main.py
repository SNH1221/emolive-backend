from fastapi import FastAPI, UploadFile, File, Form
import requests
import os
import base64

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = "bhadresh-savani/bert-base-uncased-emotion"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"
FACE_MODEL = "dima806/facial_emotions_image_detection"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

SARCASM_FLIP = {
    "joy": "disgust",
    "disgust": "joy",
    "anger": "joy",
    "sadness": "joy",
    "fear": "neutral",
    "surprise": "neutral",
    "neutral": "neutral"
}

@app.get("/")
def root():
    return {"message": "EmoLive Backend Running"}

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    # Step 1 — Sarcasm check
    sarcasm_url = f"https://router.huggingface.co/hf-inference/models/{SARCASM_MODEL}"
    sarcasm_response = requests.post(
        sarcasm_url,
        headers=HEADERS,
        json={"inputs": text}
    )
    sarcasm_raw = sarcasm_response.json()
    print(f"Sarcasm raw response: {sarcasm_raw}")

    is_sarcastic = False
    try:
        if isinstance(sarcasm_raw, list):
            sarcasm_data = sarcasm_raw[0] if isinstance(sarcasm_raw[0], list) else sarcasm_raw
            for item in sarcasm_data:
                if item["label"].lower() in ["irony", "sarcasm"] and item["score"] > 0.6:
                    is_sarcastic = True
                    break
    except:
        is_sarcastic = False

    print(f"Is sarcastic: {is_sarcastic}")

    # Step 2 — Emotion detect
    emotion_url = f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
    emotion_response = requests.post(
        emotion_url,
        headers=HEADERS,
        json={
            "inputs": text,
            "parameters": {"top_k": 7}
        }
    )

    raw = emotion_response.json()
    print(f"Emotion raw response: {raw}")  # DEBUG

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

        if is_sarcastic:
            top_label = all_scores[0]["label"]
            flipped_label = SARCASM_FLIP.get(top_label, top_label)
            original_top_score = all_scores[0]["score"]
            for e in all_scores:
                if e["label"] == flipped_label:
                    e["score"] = original_top_score
                elif e["label"] == top_label:
                    e["score"] = 0.01
            all_scores.sort(key=lambda x: x["score"], reverse=True)

        return {
            "text": text,
            "emotions": [all_scores],
            "is_sarcastic": is_sarcastic
        }

    # Fallback — raw jo bhi hai usse handle karo
    print(f"Fallback triggered — raw: {raw}")
    return {
        "text": text,
        "emotions": [[{"label": e, "score": 0.0} for e in ALL_EMOTIONS]],
        "is_sarcastic": False
    }

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
