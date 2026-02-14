from fastapi import FastAPI, Form
from transformers import pipeline

app = FastAPI()

# Load text emotion model
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

@app.get("/")
def home():
    return {"message": "EmoLive Backend Running 🚀"}

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):
    result = emotion_pipeline(text)

    return {
        "text": text,
        "emotions": result[0]
    }
