import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from huggingface_hub import InferenceClient
import ollama

# Load environment variables from .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="Persian Cooking Multi-LLM Chatbot")

# --- API CLIENTS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(api_key=HF_TOKEN)

SYSTEM_PROMPT = (
    "You are an expert Persian chef and culinary guide. "
    "Provide authentic, clear, and practical advice on Persian recipes, "
    "spices (saffron, sumac, dried limes), and techniques (such as tahdig and khoresh)."
)

# Discover active Groq models dynamically
_groq_models = []
try:
    _models_resp = groq_client.models.list().data
    _groq_models = [m.id for m in _models_resp if not any(x in m.id.lower() for x in ["whisper", "guard", "prompt", "rerank", "embed"])]
except Exception:
    pass

_groq_primary = _groq_models[0] if _groq_models else "llama-3.3-70b-versatile"
_groq_secondary = _groq_models[1] if len(_groq_models) > 1 else _groq_primary

# Supported models: Groq, Hugging Face, and Local Ollama
MODEL_MAP = {
    "ollama-local":   ("ollama", "llama3.2"),
    "groq-primary":   ("groq",   _groq_primary),
    "groq-secondary": ("groq",   _groq_secondary),
    "hf-qwen":        ("hf",     "Qwen/Qwen2.5-72B-Instruct"),
    "hf-llama":       ("hf",     "meta-llama/Llama-3.1-8B-Instruct"),
}

class ChatRequest(BaseModel):
    user_message: str
    model_choice: str = "ollama-local"

@app.get("/")
def root():
    return {
        "message": "Persian Cooking Multi-LLM Chatbot API is running!",
        "interactive_docs": "/docs",
        "chat_endpoint": "POST /chat",
        "models_endpoint": "GET /models",
        "available_models": list(MODEL_MAP.keys()),
    }

@app.get("/models")
def get_available_models():
    return {"available_models": list(MODEL_MAP.keys())}

@app.post("/chat")
def chat(request: ChatRequest):
    if request.model_choice not in MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model_choice}'. Choose from: {list(MODEL_MAP.keys())}"
        )

    provider, backend_model = MODEL_MAP[request.model_choice]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.user_message},
    ]

    try:
        if provider == "ollama":
            # Direct local inference via Ollama daemon (localhost:11434)
            response = ollama.chat(
                model=backend_model,
                messages=messages,
                options={"temperature": 0.7}
            )
            reply = response.message.content

        elif provider == "groq":
            try:
                completion = groq_client.chat.completions.create(
                    model=backend_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=600,
                )
            except Exception as groq_err:
                if ("model_not_found" in str(groq_err) or "does not exist" in str(groq_err)) and backend_model != _groq_primary:
                    backend_model = _groq_primary
                    completion = groq_client.chat.completions.create(
                        model=backend_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=600,
                    )
                else:
                    raise groq_err
            reply = completion.choices[0].message.content or getattr(completion.choices[0].message, "reasoning", "")

        elif provider == "hf":
            completion = hf_client.chat.completions.create(
                model=backend_model,
                messages=messages,
                temperature=0.7,
                max_tokens=600,
            )
            reply = completion.choices[0].message.content

        return {
            "reply": reply,
            "model_used": request.model_choice,
            "backend_model": backend_model
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))