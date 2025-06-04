import logging
import fitz  # PyMuPDF
from typing import List

from app.config import CHUNK_SIZE

logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts all text content from a PDF provided as bytes.
    """
    try:
        logger.info("Starting PDF text extraction.")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        logger.info(f"Successfully extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}", exc_info=True)
        raise RuntimeError(f"Failed to process PDF content: {e}") from e

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits a given text into smaller chunks of a specified size.
    """
    if not text:
        return []
    logger.info(
        f"Chunking text of length {len(text)} into chunks of size {chunk_size}."
    )
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Text divided into {len(chunks)} chunks.")
    return chunks