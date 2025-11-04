"""
Centralized model loading for NewsSure.
Ensures heavy ML models (transformers, embeddings, etc.) are loaded only once.
Optimized for Render deployment (low memory).
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import google.generativeai as genai

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Safely read environment variables
GEMINI_API = os.getenv("GEMINI_API")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() == "true"

# ------------------------------------------------------------
# Configure Gemini API
# ------------------------------------------------------------
if GEMINI_API:
    genai.configure(api_key=GEMINI_API)
else:
    print("⚠️ Warning: GEMINI_API key not found in environment variables!")

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# ------------------------------------------------------------
# Singleton cache
# ------------------------------------------------------------
_models = {
    "embedding": None,
    "classifier": None,
    "gemini": None
}

# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
def get_embedding_model(model_name="intfloat/multilingual-e5-small", model_dir="models"):
    if _models["embedding"] is None:
        model_path = os.path.join(model_dir, model_name.replace("/", "_"))
        print(f"🔹 Loading embedding model: {model_path if os.path.exists(model_path) else model_name}")
        _models["embedding"] = SentenceTransformer(model_path if os.path.exists(model_path) else model_name)
    return _models["embedding"]

# ------------------------------------------------------------
# Text classifier (RoBERTa MNLI)
# ------------------------------------------------------------
def get_classifier_model():
    if _models["classifier"] is None:
        print("🔹 Loading RoBERTa classifier model (text entailment)")
        _models["classifier"] = pipeline("text-classification", model="roberta-large-mnli")
    return _models["classifier"]

# ------------------------------------------------------------
# Gemini model
# ------------------------------------------------------------
def get_gemini_model():
    if _models["gemini"] is None:
        print("🔹 Initializing Gemini model")
        from google.generativeai import GenerativeModel
        _models["gemini"] = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _models["gemini"]

# ------------------------------------------------------------
# Auto-preload all models (optional, safe for Render)
# ------------------------------------------------------------
RUNNING_IN_RENDER = os.getenv("RENDER") == "true"

if PRELOAD_MODELS and not RUNNING_IN_RENDER:
    print("🚀 Preloading all models (standalone/local mode)...")
    get_embedding_model()
    get_classifier_model()
    get_gemini_model()
    print("✅ All models preloaded successfully.")
else:
    print("⚙️ Running on Render or preload disabled — models will load lazily when needed.")
