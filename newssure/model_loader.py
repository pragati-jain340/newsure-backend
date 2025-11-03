"""
Centralized model loading for TruthScope.
Ensures heavy ML models (transformers, embeddings, etc.) are loaded only once.
"""

# pip install python-dotenv

from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Go one directory up (from newssure → Backend)

from transformers import pipeline
import google.generativeai as genai
import os

# ------------------------------------------------------------
# Load .env environment variables
# ------------------------------------------------------------
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# Read from .env
GEMINI_API = os.getenv("GEMINI_API")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() == "true"


# Configure Gemini API
if GEMINI_API:
    genai.configure(api_key=GEMINI_API)
else:
    print("⚠️ Warning: GEMINI_API key not found in environment variables!")

GEMINI_MODEL_NAME = "gemini-2.5-flash"

# ------------------------------------------------------------
# Singleton Cache
# ------------------------------------------------------------
_models = {
    "embedding": None,
    "classifier": None,
    "summarizer": None,
    "gemini": None
}


# ------------------------------------------------------------
# Embedding Model (SentenceTransformer)
# ------------------------------------------------------------
def get_embedding_model(model_name="intfloat/multilingual-e5-small", model_dir="models"):
    if _models["embedding"] is None:
        model_path = os.path.join(model_dir, model_name.replace("/", "_"))
        print(f"🔹 Loading embedding model: {model_path if os.path.exists(model_path) else model_name}")
        _models["embedding"] = SentenceTransformer(model_path if os.path.exists(model_path) else model_name)
    return _models["embedding"]


# ------------------------------------------------------------
# Text Classifier (RoBERTa MNLI)
# ------------------------------------------------------------
def get_classifier_model():
    if _models["classifier"] is None:
        print("🔹 Loading RoBERTa classifier model (text entailment)")
        _models["classifier"] = pipeline("text-classification", model="roberta-large-mnli")
    return _models["classifier"]


# ------------------------------------------------------------
# Local Summarizer (BART)
# ------------------------------------------------------------
def get_summarizer_model():
    if _models["summarizer"] is None:
        print("🔹 Loading BART summarization model")
        _models["summarizer"] = pipeline("summarization", model="facebook/bart-large-cnn")
    return _models["summarizer"]


# ------------------------------------------------------------
# Gemini Model (Google Generative AI)
# ------------------------------------------------------------
def get_gemini_model():
    if _models["gemini"] is None:
        print("🔹 Initializing Gemini model")
        from google.generativeai import GenerativeModel
        _models["gemini"] = genai.GenerativeModel(GEMINI_MODEL_NAME)
    return _models["gemini"]


# ------------------------------------------------------------
# Auto-preload all models (optional)
# ------------------------------------------------------------
if __name__ == "__main__" or PRELOAD_MODELS:
    get_embedding_model()
    get_classifier_model()
    get_summarizer_model()
    get_gemini_model()
    print("✅ All models preloaded successfully.")
