from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import os

TINY_LLAMA_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL_ID = "all-MiniLM-L6-v2"

def download_models():
    try:
        print(f"📥 Downloading tokenizer and model: {TINY_LLAMA_ID}")
        tokenizer = AutoTokenizer.from_pretrained(TINY_LLAMA_ID, cache_dir="./models/tokenizer")
        model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA_ID, cache_dir="./models/tinyllama")
        print("✅ TinyLlama model and tokenizer downloaded.")

        print(f"📥 Downloading embedding model: {EMBEDDING_MODEL_ID}")
        embed_model = SentenceTransformer(EMBEDDING_MODEL_ID, cache_folder="./models/embeddings")
        print("✅ Embedding model downloaded.")

        print("🎉 All models downloaded and cached locally.")
    except Exception as e:
        print(f"❌ Error during model download: {e}")

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    download_models()

