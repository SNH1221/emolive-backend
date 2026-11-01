
from fastapi import FastAPI, Form
import requests
import os
import json

app = FastAPI()

# ==========================================
# CONFIGURATION
# ==========================================

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TEXT_MODEL = "SamLowe/roberta-base-go_emotions"
SARCASM_MODEL = "cardiffnlp/twitter-roberta-base-irony"

GROQ_MODEL = "openai/gpt-oss-20b"

TIMEOUT = 60

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ==========================================
# GENUINE POSITIVE DETECTION
# ==========================================

GENUINE_POSITIVE = [
    "got the job", "i love you", "best day", "so happy",
    "congratulations", "got promoted", "we won", "i passed",
    "she said yes", "he said yes", "i got in", "accepted",
    "i am so excited", "can't wait", "finally did it",
    "i can't believe i got", "best news",
    "everything is normal", "doctor said",
    "test results normal", "all clear", "healthy",
    "recovered", "feeling better"
]

NEGATIVE_CONTEXT = [
    "ignored", "hate", "stuck", "terrible", "awful",
    "worst", "again", "another monday", "as usual",
    "obviously", "totally fine", "just great",
    "love being", "love getting", "love waiting",
    "love sitting", "stealing", "talking over",
    "ghost", "invisible"
]

ALLOWED_LABELS = {
    "admiration", "amusement", "anger", "annoyance",
    "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "neutral",
    "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise"
}


# ==========================================
# GROQ API HELPER
# ==========================================

def call_groq(
    prompt,
    system_prompt,
    max_tokens=1200,
    temperature=0.1,
    json_mode=False
):

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing from environment variables."
        )

    payload = {
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
        "max_completion_tokens": max_tokens,
        "include_reasoning": False
    }

    if json_mode:
        payload["response_format"] = {
            "type": "json_object"
        }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",

        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },

        json=payload,
        timeout=TIMEOUT
    )

    print("\n========== GROQ API DEBUG ==========")
    print("Model:", GROQ_MODEL)
    print("Status Code:", response.status_code)
    print("Full Response:", response.text)
    print("====================================\n")

    response.raise_for_status()

    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise ValueError(
            f"Groq returned no choices: {data}"
        )

    choice = data["choices"][0]

    print("Finish Reason:", choice.get("finish_reason"))

    if choice.get("finish_reason") == "length":
        raise ValueError(
            "Groq response was truncated due to token limit."
        )

    message = choice.get("message", {})
    content = message.get("content")

    if not content:
        raise ValueError(
            f"Groq returned empty content: {data}"
        )

    return content.strip()


# ==========================================
# HUGGING FACE API HELPER
# ==========================================

def call_huggingface(model, text, parameters=None):

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is missing from environment variables."
        )

    url = (
        f"https://router.huggingface.co/"
        f"hf-inference/models/{model}"
    )

    payload = {
        "inputs": text
    }

    if parameters:
        payload["parameters"] = parameters

    response = requests.post(
        url,
        headers=HF_HEADERS,
        json=payload,
        timeout=TIMEOUT
    )

    print("\n========== HUGGING FACE DEBUG ==========")
    print("Model:", model)
    print("Status:", response.status_code)
    print("Response:", response.text)
    print("=======================================\n")

    response.raise_for_status()

    return response.json()


# ==========================================
# RESPONSE NORMALIZATION
# ==========================================

def extract_scores(raw):

    if not isinstance(raw, list) or not raw:
        raise ValueError(
            f"Invalid Hugging Face response: {raw}"
        )

    # Handle nested model response
    emotions = raw[0] if isinstance(raw[0], list) else raw

    if not emotions or not isinstance(emotions, list):
        raise ValueError("No emotion scores found.")

    scores = []

    for item in emotions:

        if not isinstance(item, dict):
            continue

        label = item.get("label")
        score = item.get("score")

        if label is None or score is None:
            continue

        scores.append({
            "label": str(label).lower(),
            "score": round(float(score), 4)
        })

    if not scores:
        raise ValueError("No valid emotion scores.")

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scores


# ==========================================
# GROQ EMOTION REFINEMENT
# ==========================================

def refine_emotions(
    text,
    raw_scores,
    is_sarcastic
):

    top_5 = raw_scores[:5]

    top_5_str = ", ".join(
        [
            f"{e['label']}({int(e['score'] * 100)}%)"
            for e in top_5
        ]
    )

    sarcasm_note = (
        "The text appears to be sarcastic or ironic."
        if is_sarcastic
        else "No sarcasm was detected."
    )

    prompt = f"""
Analyze the emotional meaning of the following text.

TEXT:
{text}

ORIGINAL NLP MODEL RESULTS:
{top_5_str}

SARCASM INFORMATION:
{sarcasm_note}

TASK:

Refine the emotion scores using the complete context and meaning.

If sarcasm is detected, consider the underlying emotion rather
than interpreting the literal positive words.

Use ONLY these emotion labels:

admiration, amusement, anger, annoyance, approval, caring,
confusion, curiosity, desire, disappointment, disapproval,
disgust, embarrassment, excitement, fear, gratitude, grief,
joy, love, nervousness, neutral, optimism, pride, realization,
relief, remorse, sadness, surprise

RULES:

1. Return exactly 7 emotions.
2. Scores must be between 0 and 1.
3. All 7 scores must sum to 1.0.
4. Put the highest score first.
5. Consider the original NLP results.
6. For Monday/work frustration, prefer annoyance over disapproval.
7. For passive-aggressive statements, prefer annoyance.
8. Do not invent emotions unrelated to the text.
9. Return only a JSON object.
10. Do not include explanations or reasoning.

REQUIRED JSON STRUCTURE:

{{
    "emotions": [
        {{"label": "fear", "score": 0.45}},
        {{"label": "nervousness", "score": 0.25}},
        {{"label": "surprise", "score": 0.15}},
        {{"label": "annoyance", "score": 0.05}},
        {{"label": "disgust", "score": 0.05}},
        {{"label": "disapproval", "score": 0.03}},
        {{"label": "confusion", "score": 0.02}}
    ]
}}
"""

    groq_text = call_groq(
        prompt=prompt,

        system_prompt=(
            "You are an expert in contextual emotion analysis. "
            "Return only a valid JSON object with an emotions array. "
            "Never include additional text."
        ),

        max_tokens=1200,
        temperature=0.1,
        json_mode=True
    )

    print("Groq JSON Output:", groq_text)

    result = json.loads(groq_text)

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    final_emotions = result.get("emotions")

    if not isinstance(final_emotions, list):
        raise ValueError("Missing emotions array.")

    if len(final_emotions) != 7:
        raise ValueError(
            f"Expected 7 emotions, got {len(final_emotions)}."
        )

    validated = []

    for item in final_emotions:

        label = str(item["label"]).lower()
        score = float(item["score"])

        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"Invalid emotion label: {label}"
            )

        if not 0 <= score <= 1:
            raise ValueError(
                f"Invalid emotion score: {score}"
            )

        validated.append({
            "label": label,
            "score": score
        })

    total = sum(
        item["score"] for item in validated
    )

    if total <= 0:
        raise ValueError("Total emotion score cannot be zero.")

    # Normalize scores to exactly 1.0
    for item in validated:
        item["score"] = round(
            item["score"] / total, 4
        )

    # Correct rounding difference
    difference = round(
        1.0 - sum(item["score"] for item in validated),
        4
    )

    validated[0]["score"] = round(
        validated[0]["score"] + difference,
        4
    )

    validated.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return validated


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "EmoLive Backend Running"
    }


# ==========================================
# STEP 1: EMOTION DETECTION
# ==========================================

@app.post("/detect-text-emotion")
async def detect_text_emotion(text: str = Form(...)):

    text_lower = text.lower()

    has_genuine_positive = any(
        phrase in text_lower
        for phrase in GENUINE_POSITIVE
    )

    has_negative_context = any(
        phrase in text_lower
        for phrase in NEGATIVE_CONTEXT
    )

    # ======================================
    # STEP 1: SARCASM DETECTION
    # ======================================

    is_sarcastic = False

    if not has_genuine_positive:

        try:

            sarcasm_raw = call_huggingface(
                SARCASM_MODEL,
                text
            )

            print("Sarcasm raw:", sarcasm_raw)

            if isinstance(sarcasm_raw, list) and sarcasm_raw:

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
                        and float(item.get("score", 0)) > threshold
                    ):

                        is_sarcastic = True
                        break

        except Exception as e:

            print(
                "Sarcasm Detection Error:",
                repr(e)
            )

            is_sarcastic = False

    print("Is sarcastic:", is_sarcastic)

    # ======================================
    # STEP 2: EMOTION CLASSIFICATION
    # ======================================

    try:

        emotion_raw = call_huggingface(
            TEXT_MODEL,
            text,
            parameters={
                "top_k": 28
            }
        )

        print("Emotion raw:", emotion_raw)

        raw_scores = extract_scores(emotion_raw)

    except Exception as e:

        print(
            "Emotion Detection Error:",
            repr(e)
        )

        raw_scores = [
            {
                "label": "neutral",
                "score": 1.0
            }
        ]

    print("Original Emotion Scores:", raw_scores)

    # ======================================
    # STEP 3: GROQ CONTEXTUAL REFINEMENT
    # ======================================

    try:

        final_emotions = refine_emotions(
            text,
            raw_scores,
            is_sarcastic
        )

        print(
            "Final Refined Emotions:",
            final_emotions
        )

    except Exception as e:

        print(
            "Groq Emotion Refinement Error:",
            repr(e)
        )

        # Fallback to Hugging Face results
        final_emotions = raw_scores

    # ======================================
    # FINAL RESPONSE
    # ======================================

    return {
        "text": text,
        "emotions": [final_emotions],
        "is_sarcastic": is_sarcastic
    }


# ==========================================
# STEP 2: EMPATHETIC AI RESPONSE
# ==========================================

@app.post("/generate-ai-response")
async def generate_ai_response(
    emotion: str = Form(...)
):

    try:

        prompt = when_emotion_prompt(emotion)

        ai_text = call_groq(
            prompt=prompt,

            system_prompt=(
                "You are a compassionate emotional support assistant. "
                "Give warm, empathetic responses in 2-3 sentences. "
                "Be natural, supportive and non-judgmental."
            ),

            max_tokens=300,
            temperature=0.7,
            json_mode=False
        )

        print(
            "AI Generated Response:",
            ai_text
        )

        return {
            "response": ai_text
        }

    except Exception as e:

        print(
            "AI Response Error:",
            repr(e)
        )

        return {
            "response": "Could not generate response."
        }


# ==========================================
# EMOTION-BASED PROMPTS
# ==========================================

def when_emotion_prompt(emotion: str) -> str:

    emotion = emotion.lower().strip()

    if emotion in [
        "joy",
        "excitement",
        "admiration"
    ]:

        return (
            f"The user is feeling {emotion}. "
            "Give a warm, celebratory empathetic response "
            "in 2-3 sentences. Suggest one way to maintain "
            "this positive energy."
        )

    elif emotion in [
        "sadness",
        "grief",
        "disappointment",
        "remorse"
    ]:

        return (
            f"The user is feeling {emotion}. "
            "Give a compassionate empathetic response "
            "in 2-3 sentences. Suggest one gentle positive reframe."
        )

    elif emotion in [
        "anger",
        "annoyance",
        "disapproval"
    ]:

        return (
            f"The user is feeling {emotion}. "
            "Give a calm, understanding empathetic response "
            "in 2-3 sentences. Suggest one way to channel "
            "this energy positively."
        )

    elif emotion in [
        "fear",
        "nervousness"
    ]:

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

    elif emotion in [
        "love",
        "caring"
    ]:

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
