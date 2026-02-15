import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DEVICE = os.getenv("DEVICE", "cpu")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    TOP_K = int(os.getenv("TOP_K", 5))
    FAISS_INDEX_PATH = "data/faiss_index/index.bin"
    EMBEDDING_DIM = 512  # CLIP base dimension

settings = Settings()