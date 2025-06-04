import logging
import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level variables to hold the client and collection
chroma_client = None
pdf_collection = None
sentence_transformer_ef = None

def initialize_db():
    """Initializes the ChromaDB client and collection."""
    global chroma_client, pdf_collection, sentence_transformer_ef
    try:
        logger.info(f"Initializing ChromaDB client with path: '{CHROMA_DB_PATH}'")
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        pdf_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=sentence_transformer_ef
        )
        logger.info(
            f"Successfully connected to ChromaDB and got/created collection '{COLLECTION_NAME}'."
        )
    except Exception as e:
        logger.critical(f"Failed to initialize ChromaDB: {e}", exc_info=True)
        raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

def get_collection():
    """Returns the initialized PDF collection."""
    if pdf_collection is None:
        logger.error("Attempted to get collection before DB initialization.")
        raise RuntimeError("Database not initialized. Call initialize_db() during app startup.")
    return pdf_collection

def close_db():
    """Placeholder for any cleanup, though PersistentClient might not need explicit close."""
    global chroma_client, pdf_collection
    if chroma_client:
        logger.info("Closing ChromaDB client (if applicable).")
        # chroma_client.close() # If there's a close method. PersistentClient might not.
    chroma_client = None
    pdf_collection = None
    logger.info("ChromaDB resources released.")