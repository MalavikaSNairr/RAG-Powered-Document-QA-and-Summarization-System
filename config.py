import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Hugging Face API Key
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# FAISS Index path
FAISS_INDEX_PATH = "faiss_index"

# Debug print (optional, remove in production)
if not HUGGINGFACE_API_KEY:
    raise ValueError("❌ Hugging Face API Key not found. Please check your .env file.")
