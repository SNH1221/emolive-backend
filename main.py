from fastapi import FastAPI, Form
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.get("/")
def home():
    return {"message": "EmoLive Backend Running"}

@app.post("/detect-text-emotion")
def detect_text_emotion(text: str = Form(...)):

    payload = {"inputs": text}

    response = requests.post(API_URL, headers=headers, json=payload)

    return {
        "text": text,
        "emotions": response.json()
    }
