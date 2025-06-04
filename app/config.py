import os
import logging
from dotenv import load_dotenv

# ===== Logger for Config =====
# This logger is specific to config loading issues, separate from app-wide logger.
config_logger = logging.getLogger("config")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ===== Environment Variable Loading =====
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    config_logger.error("GEMINI_API_KEY not found in environment variables. AI features will not work.")
    # Depending on strictness, you might raise an error:
    # raise ValueError("GEMINI_API_KEY not found. Application cannot start.")

# ===== Constants =====
CHUNK_SIZE: int = 1000
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHROMA_DB_PATH: str = "./chroma_db_store"  # Changed path to avoid conflict if run from root
COLLECTION_NAME: str = "pdf_docs"
GEMINI_MODEL_NAME: str = "gemini-1.5-flash"