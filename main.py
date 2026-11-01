
from fastapi import FastAPI, Form
import requests
import os
import json

app = FastAPI()

# API KEYS
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# MODELS
TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"
GROQ_MODEL = "openai/gpt-oss-20b"

# API CONFIGURATION
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

TIMEOUT = 60

GENUINE_POSITIVE = [
    "got the job", "i love you", "best day", "so happy",
    "congratulations", "got promoted", "we won", "i passed",
    "she said yes", "he said yes", "i got in", "accepted",
    "i am so excited", "can't wait", "finally did it",
    "i can't believe i got", "best news", "everything is normal",
    "doctor said", "test results normal", "all clear",
    "healthy", "recovered", "feeling better"
]

NEGATIVE_CONTEXT = [
    "ignored", "hate", "stuck", "terrible", "awful", "worst",
    "again", "another monday", "as usual", "obviously",
    "totally fine", "just great", "love being", "love getting",
    "love waiting", "love sitting", "stealing", "talking over",
    "ghost", "invisible"
]


# ==========================================
# GROQ API HELPER
# ==========================================

def call_groq(prompt, system_prompt, max_tokens=500, temperature=0.1):

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing in Render environment variables.")

    groq_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",

        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },

        json={
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        },

        timeout=TIMEOUT
    )

    # DEBUGGING INFORMATION
    print("========== GROQ API DEBUG ==========")
    print("Groq Model:", GROQ_MODEL)
    print("Groq Status Code:", groq_response.status_code)
    print("Groq Full Response:", groq_response.text)
    print("====================================")

    # Check HTTP errors
    groq_response.raise_for_status()

    groq_data = groq_response.json()

    # Check whether choices exists
    if "choices" not in groq_data:
        raise ValueError(f"Groq API returned no choices: {groq_data}")

    if not groq_data["choices"]:
        raise ValueError("Groq returned an empty choices list.")

    message = groq_data["choices"][0].get("message", {})
    content = message.get("content")

    if not content:
        raise ValueError(f"Groq returned empty content: {groq_data}")

    return content.strip()


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():
    return {
        "message": "EmoLive Backend Running"
    }


# ==========================================
# EMOTION DETECTION
# ==========================================

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    text_lower = text.lower()

    has_genuine_positive = any(
        phrase in text_lower for phrase in GENUINE_POSITIVE
    )

    has_negative_context = any(
        phrase in text_lower for phrase in NEGATIVE_CONTEXT
    )

    # ======================================
    # STEP 1: SARCASM DETECTION
    # ======================================

    is_sarcastic = False

    if not has_genuine_positive:

        try:

            if not HF_TOKEN:
                raise ValueError("HF_TOKEN is missing.")

            sarcasm_url = (
                f"https://router.huggingface.co/hf-inference/models/{SARCASM_MODEL}"
            )

            sarcasm_response = requests.post(
                sarcasm_url,
                headers=HEADERS,
                json={"inputs": text},
                timeout=TIMEOUT
            )

            print("========== HUGGING FACE SARCASM ==========")
            print("Status:", sarcasm_response.status_code)
            print("Response:", sarcasm_response.text)

            sarcasm_response.raise_for_status()

            sarcasm_raw = sarcasm_response.json()

            print("Sarcasm raw:", sarcasm_raw)

            if isinstance(sarcasm_raw, list) and len(sarcasm_raw) > 0:

                sarcasm_data = (
                    sarcasm_raw[0]
                    if isinstance(sarcasm_raw[0], list)
                    else sarcasm_raw
                )

                for item in sarcasm_data:

                    threshold = (
                        0.75 if has_negative_context else 0.85
                    )

                    if (
                        item.get("label", "").lower()
                        in ["irony", "sarcasm"]
                        and item.get("score", 0) > threshold
                    ):

                        is_sarcastic = True
                        break

        except Exception as e:

            print("Sarcasm Detection Error:", repr(e))
            is_sarcastic = False

    print("Is sarcastic:", is_sarcastic)

    # ======================================
    # STEP 2: EMOTION CLASSIFICATION
    # ======================================

    try:

        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is missing.")

        emotion_url = (
            f"https://router.huggingface.co/hf-inference/models/{TEXT_MODEL}"
        )

        emotion_response = requests.post(
            emotion_url,
            headers=HEADERS,
            json={
                "inputs": text,
                "parameters": {
                    "top_k": 28
                }
            },
            timeout=TIMEOUT
        )

        print("========== HUGGING FACE EMOTION ==========")
        print("Status:", emotion_response.status_code)
        print("Response:", emotion_response.text)

        emotion_response.raise_for_status()

        raw = emotion_response.json()

        print("Emotion raw:", raw)

        if isinstance(raw, list) and len(raw) > 0:

            emotions_raw = (
                raw[0]
                if isinstance(raw[0], list)
                else raw
            )

            raw_scores = [
                {
                    "label": e["label"].lower(),
                    "score": round(e["score"], 4)
                }
                for e in emotions_raw
            ]

            raw_scores.sort(
                key=lambda x: x["score"],
                reverse=True
            )

        else:

            raise ValueError(
                f"Unexpected Hugging Face emotion response: {raw}"
            )

    except Exception as e:

        print("Emotion Detection Error:", repr(e))

        raw_scores = [
            {
                "label": "neutral",
                "score": 1.0
            }
        ]

    # ======================================
    # STEP 3: GROQ CONTEXTUAL REFINEMENT
    # ======================================

    try:

        top_5 = raw_scores[:5]

        top_5_str = ", ".join(
            [
                f"{e['label']}({int(e['score'] * 100)}%)"
                for e in top_5
            ]
        )

        sarcasm_note = (
            "The text appears to be sarcastic/ironic."
            if is_sarcastic else ""
        )

        prompt = f"""
You are an emotion detection expert.
Analyze this text and return corrected emotion scores.

Text: "{text}"

NLP Model detected (top 5): {top_5_str}

{sarcasm_note}

Instructions:

- Consider the full context and meaning.
- If sarcastic, adjust emotions accordingly.
- Return ONLY a JSON array of top 7 emotions with corrected scores.
- Use only these labels:

admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, neutral, optimism, pride, realization,
relief, remorse, sadness, surprise

- Scores must sum to 1.0.
- For Monday/work frustration, prefer annoyance over disapproval.
- For passive aggressive statements, prefer annoyance.
- Return ONLY valid JSON array.
- No markdown.
- No explanation.

Example:

[
    {{"label": "fear", "score": 0.85}},
    {{"label": "nervousness", "score": 0.10}},
    {{"label": "surprise", "score": 0.05}}
]
"""

        groq_text = call_groq(
            prompt=prompt,

            system_prompt=(
                "You are an emotion detection expert. "
                "Always respond with valid JSON array only, "
                "no explanation."
            ),

            max_tokens=500,
            temperature=0.1
        )

        # Clean JSON response
        groq_text = (
            groq_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        print("Groq parsed:", groq_text)

        final_emotions = json.loads(groq_text)

        if not isinstance(final_emotions, list) or not final_emotions:
            raise ValueError("Groq returned invalid emotion array.")

        # Validate labels and scores
        allowed_labels = {
            "admiration", "amusement", "anger", "annoyance",
            "approval", "caring", "confusion", "curiosity",
            "desire", "disappointment", "disapproval", "disgust",
            "embarrassment", "excitement", "fear", "gratitude",
            "grief", "joy", "love", "nervousness", "neutral",
            "optimism", "pride", "realization", "relief",
            "remorse", "sadness", "surprise"
        }

        for emotion in final_emotions:

            if emotion["label"].lower() not in allowed_labels:
                raise ValueError(
                    f"Invalid emotion label: {emotion['label']}"
                )

            emotion["label"] = emotion["label"].lower()
            emotion["score"] = float(emotion["score"])

            if not 0 <= emotion["score"] <= 1:
                raise ValueError("Invalid emotion score.")

        final_emotions.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        print("Final emotions:", final_emotions)

        return {
            "text": text,
            "emotions": [final_emotions],
            "is_sarcastic": is_sarcastic
        }

    except Exception as e:

        print("Groq Emotion Refinement Error:", repr(e))

        # Fallback to original NLP model scores
        return {
            "text": text,
            "emotions": [raw_scores],
            "is_sarcastic": is_sarcastic
        }


# ==========================================
# AI EMPATHETIC RESPONSE GENERATION
# ==========================================

@app.post("/generate-ai-response")
async def generate_ai_response(emotion: str = Form(...)):

    try:

        prompt = when_emotion_prompt(emotion)

        ai_text = call_groq(
            prompt=prompt,

            system_prompt=(
                "You are a compassionate emotional support assistant. "
                "Give warm, empathetic responses in 2-3 sentences."
            ),

            max_tokens=200,
            temperature=0.7
        )

        print("AI Generated Response:", ai_text)

        return {
            "response": ai_text
        }

    except Exception as e:

        print("AI Response Error:", repr(e))

        return {
            "response": "Could not generate response."
        }


# ==========================================
# EMOTION-BASED PROMPTS
# ==========================================

def when_emotion_prompt(emotion: str) -> str:

    emotion = emotion.lower()

    if emotion in ["joy", "excitement", "admiration"]:

        return (
            f"The user is feeling {emotion}. "
            "Give a warm, celebratory empathetic response "
            "in 2-3 sentences. Suggest one way to maintain "
            "this positive energy."
        )

    elif emotion in ["sadness", "grief", "disappointment", "remorse"]:

        return (
            f"The user is feeling {emotion}. "
            "Give a compassionate empathetic response "
            "in 2-3 sentences. Suggest one gentle positive reframe."
        )

    elif emotion in ["anger", "annoyance", "disapproval"]:

        return (
            f"The user is feeling {emotion}. "
            "Give a calm, understanding empathetic response "
            "in 2-3 sentences. Suggest one way to channel "
            "this energy positively."
        )

    elif emotion in ["fear", "nervousness"]:

        return (
            f"The user is feeling {emotion}. "
            "Give a reassuring empathetic response "
            "in 2-3 sentences. Suggest one grounding technique."
        )

    elif emotion == "relief":

        return (
            "The user is feeling relief. "
            "Give a warm, validating empathetic response "
            "in 2-3 sentences."
        )

    elif emotion in ["love", "caring"]:

        return (
            f"The user is feeling {emotion}. "
            "Give a warm, heartfelt empathetic response "
            "in 2-3 sentences."
        )

    elif emotion == "confusion":

        return (
            "The user is feeling confused. "
            "Give a calm, clarifying empathetic response "
            "in 2-3 sentences."
        )

    elif emotion == "surprise":

        return (
            "The user is feeling surprised. "
            "Give a fun, engaging empathetic response "
            "in 2-3 sentences."
        )

    else:

        return (
            f"The user is feeling {emotion}. "
            "Give a gentle, encouraging empathetic response "
            "in 2-3 sentences."
        )
