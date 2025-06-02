
import logging
import os
import uuid
from typing import List, Dict, Any

import chromadb
import fitz  # PyMuPDF
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ===== Constants & Configuration =====
CHUNK_SIZE: int = 1000
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
CHROMA_DB_PATH: str = "./chroma_db"
COLLECTION_NAME: str = "pdf_docs"
GEMINI_MODEL_NAME: str = "gemini-1.5-flash" # Adjusted to a commonly available model, check Gemini docs for the latest


# ===== Logger Setup =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ===== Environment Variable Loading =====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    # Depending on the application, you might want to raise an error here or exit.
    # For now, we'll let genai.configure fail if the key is missing.

# ===== FastAPI Application Initialization =====
app = FastAPI(
    title="PDF QA Assistant Pro",
    description="Upload PDFs and ask questions using generative AI.",
    version="1.0.0",
)

# ===== AI and Database Initialization =====
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        raise ValueError("GEMINI_API_KEY is not set.")

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    pdf_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=sentence_transformer_ef
    )
    logger.info(
        f"Successfully initialized and connected to ChromaDB at '{CHROMA_DB_PATH}' "
        f"with collection '{COLLECTION_NAME}'."
    )
    logger.info(f"Gemini AI configured with model: '{GEMINI_MODEL_NAME}'.")

except Exception as e:
    logger.critical(f"Failed to initialize AI or Database: {e}", exc_info=True)
    # Depending on deployment, might exit or have a degraded mode
    raise RuntimeError(f"Core service initialization failed: {e}") from e


# ===== Helper Functions =====
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts all text content from a PDF provided as bytes.

    Args:
        pdf_bytes: The byte content of the PDF file.

    Returns:
        A string containing all extracted text.

    Raises:
        RuntimeError: If PDF processing fails.
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


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Splits a given text into smaller chunks of a specified size.

    Args:
        text: The text string to be chunked.
        chunk_size: The maximum size of each chunk.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []
    logger.info(
        f"Chunking text of length {len(text)} into chunks of size {chunk_size}."
    )
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Text divided into {len(chunks)} chunks.")
    return chunks


# ===== HTML Content for the Root Route =====
# For larger applications, consider using Jinja2Templates.
HTML_FOR_HOME_ROUTE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ PDF QA Assistant Pro ✨</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        :root {
            --primary-color: #2563eb; --primary-hover-color: #1d4ed8; --secondary-color: #059669;
            --background-color: #f3f4f6; --card-background-color: #ffffff; --text-color: #1f2937;
            --muted-text-color: #4b5563; --border-color: #d1d5db; --success-color: #10b981;
            --font-family: 'Poppins', sans-serif;
        }
        body {
            font-family: var(--font-family); background-color: var(--background-color); padding: 2em;
            color: var(--text-color); line-height: 1.6; display: flex; flex-direction: column;
            align-items: center; min-height: 100vh; margin: 0;
        }
        .container {
            width: 100%; max-width: 700px; background-color: var(--card-background-color); padding: 2em;
            border-radius: 12px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1); margin-bottom: 2em;
        }
        h2 {
            color: var(--primary-color); margin-top: 0; margin-bottom: 1em; font-weight: 600;
            border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5em; display: flex; align-items: center;
        }
        h2 .fas { margin-right: 0.5em; }
        form {
            background-color: #f9fafb; padding: 1.5em; border-radius: 8px; border: 1px solid var(--border-color);
            margin-bottom: 2em; box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        }
        label { display: block; margin-bottom: 0.5em; font-weight: 500; color: var(--muted-text-color); }
        input[type="file"], input[type="text"], input[type="number"] {
            padding: 0.75em 1em; margin-bottom: 1em; border: 1px solid var(--border-color);
            border-radius: 6px; width: 100%; box-sizing: border-box; font-size: 0.95em;
            color: var(--text-color); background-color: var(--card-background-color);
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        input[type="file"] { padding: 0.5em; }
        input[type="file"]::file-selector-button {
            background-color: var(--secondary-color); color: white; padding: 0.6em 1em; border: none;
            border-radius: 4px; cursor: pointer; font-weight: 500; margin-right: 1em;
            transition: background-color 0.2s ease;
        }
        input[type="file"]::file-selector-button:hover { background-color: #047857; }
        input:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2); outline: none; }
        button {
            background-color: var(--primary-color); color: white; padding: 0.8em 1.5em; border: none;
            border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 1em; margin-top: 0.5em;
            transition: background-color 0.2s ease, transform 0.1s ease;
            display: inline-flex; align-items: center; gap: 0.5em;
        }
        button:hover { background-color: var(--primary-hover-color); transform: translateY(-2px); }
        button:active { transform: translateY(0px); }
        #answer-box {
            margin-top: 2em; border-top: 3px solid var(--success-color); padding-top: 1.5em;
            background-color: #f0fdf4; border-radius: 8px; padding: 1.5em;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05); color: var(--text-color);
        }
        #answer-box h3 { color: var(--success-color); margin-top: 0; font-weight: 600; display: flex; align-items: center; }
        #answer-box h3 .fas { margin-right: 0.5em; }
        #answer-box p { margin-bottom: 0; font-size: 0.95em; line-height: 1.7; }
        .input-group { margin-bottom: 1em; }
        .input-group label { display: block; font-size: 0.9em; margin-bottom: 0.3em; color: var(--muted-text-color); }
    </style>
    <script>
    async function askQuestion(event) {
        event.preventDefault();
        const form = event.target;
        const question = form.question.value;
        const k_results = form.k_results.value;
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ question, k_results })
        });
        const data = await response.json();
        document.getElementById("answer-box").innerHTML = `<h3>📌 Answer:</h3><p>${data.answer}</p>`;
    }
    function handleFileUpload(event) {
        const uploadStatus = document.getElementById('upload-status');
        if (uploadStatus) {
             uploadStatus.innerHTML = `<p style="color: var(--primary-color);"><i class="fas fa-spinner fa-spin"></i> Uploading PDF...</p>`;
        }
    }
    </script>
</head>
<body>
    <div class="container">
        <h2><i class="fas fa-file-pdf"></i> PDF Power-Up</h2>
        
        <form action="/upload-pdf" enctype="multipart/form-data" method="post" onsubmit="handleFileUpload(event)">
             <div class="input-group">
                <label for="file-upload"><i class="fas fa-upload"></i> Select PDF Document</label>
                <input type="file" name="file" id="file-upload" required>
            </div>
            <button type="submit"><i class="fas fa-cloud-upload-alt"></i> Upload PDF</button>
            <div id="upload-status" style="margin-top: 0.5em;"></div>
        </form>

        <h2><i class="fas fa-question-circle"></i> Ask Your Question</h2>
        <form onsubmit="askQuestion(event)">
            <div class="input-group">
                <label for="question-input"><i class="fas fa-keyboard"></i> Enter your question</label>
                <input type="text" id="question-input" name="question" placeholder="e.g., What are the main conclusions?" required>
            </div>
            <div class="input-group">
                <label for="k_results-input"><i class="fas fa-list-ol"></i> Number of results to consider (k)</label>
                <input type="number" id="k_results-input" name="k_results" value="3" min="1" max="10">
            </div>
            <button type="submit"><i class="fas fa-paper-plane"></i> Ask</button>
        </form>

        <div id="answer-box">
            <!-- Answer will appear here -->
            <h3><i class="fas fa-info-circle"></i> Waiting for question...</h3>
            <p>Upload a PDF and ask a question to see the answer here.</p>
        </div>
    </div>
</body>
</html>
"""


# ===== API Routes =====
@app.get("/", response_class=HTMLResponse)
async def get_home_page():
    """Serves the main HTML page for the PDF QA Assistant."""
    return HTMLResponse(content=HTML_FOR_HOME_ROUTE)


@app.post("/upload-pdf", response_class=HTMLResponse)
async def handle_pdf_upload(file: UploadFile = File(...)):
    """
    Handles PDF file uploads, extracts text, chunks it, and stores embeddings in ChromaDB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name provided.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        logger.info(f"Received PDF upload: {file.filename}")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        text_content = extract_text_from_pdf(pdf_bytes)
        if not text_content.strip():
            logger.warning(f"No text could be extracted from PDF: {file.filename}")
            # Return a user-friendly message but don't store if no text
            return HTMLResponse(
                content=f"<p>⚠️ No text content found in <strong>{file.filename}</strong>. "
                        f"The file might be image-based or password-protected.</p><a href='/'>🔙 Back</a>",
                status_code=200 # Or 400 if this is considered a client error
            )

        text_chunks = chunk_text(text_content)

        # Generate unique IDs for each chunk to avoid collisions
        document_id = str(uuid.uuid4())
        chunk_ids = [f"{document_id}_{i}" for i, _ in enumerate(text_chunks)]
        # Use filename as metadata for all chunks from this document
        chunk_metadatas = [{"source_document": file.filename}] * len(text_chunks)

        pdf_collection.add(
            documents=text_chunks, ids=chunk_ids, metadatas=chunk_metadatas
        )

        logger.info(
            f"Successfully stored {len(text_chunks)} chunks from {file.filename} "
            f"(Document ID: {document_id}) into ChromaDB."
        )
        # Provide feedback to the user
        return HTMLResponse(
            content=f"<p>✅ Successfully uploaded and processed <strong>{file.filename}</strong>. "
            f"{len(text_chunks)} text chunks stored.</p><a href='/'>🔙 Back</a>"
        )

    except RuntimeError as e:
        logger.error(f"PDF processing error during upload of {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error processing PDF: {e}")
    except chromadb.errors.ChromaError as e:
        logger.error(f"ChromaDB error during upload of {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error during PDF processing.")
    except Exception as e:
        logger.error(f"Unexpected error during PDF upload ({file.filename}): {e}", exc_info=True)
        # For unhandled exceptions, send a generic error to client.
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.")
    finally:
        await file.close()


@app.post("/ask", response_class=JSONResponse)
async def handle_ask_question(
    question: str = Form(...), k_results: int = Form(3)
) -> JSONResponse:
    """
    Receives a question, queries ChromaDB for relevant context,
    and uses Gemini AI to generate an answer.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not (1 <= k_results <= 10): # Assuming max 10 is a reasonable limit
        raise HTTPException(status_code=400, detail="k_results must be between 1 and 10.")

    try:
        logger.info(
            f"Received question: '{question}' (Querying for top {k_results} results)."
        )

        query_results = pdf_collection.query(
            query_texts=[question], n_results=k_results
        )

        if not query_results or not query_results.get("documents") or not query_results["documents"][0]:
            logger.warning(f"No relevant documents found in ChromaDB for question: '{question}'")
            return JSONResponse(
                content={"answer": "Sorry, I couldn't find any relevant information in the uploaded documents to answer your question."},
                status_code=200 # Or 404 if "no data" should be treated as an error
            )

        # Consolidate documents to form the context
        context_parts = query_results["documents"][0]
        context = "\n\n---\n\n".join(context_parts) # Use a clear separator

        # Construct a clear prompt for the generative model
        prompt = (
            f"Based on the following context from PDF documents, please answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        logger.debug(f"Prompt sent to Gemini: {prompt[:500]}...") # Log truncated prompt

        # Initialize the Gemini model (consider making this reusable if state allows)
        generative_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        ai_response = await generative_model.generate_content_async(prompt) # Use async version

        logger.info(f"Answer generated successfully for question: '{question}'")
        return JSONResponse(content={"answer": ai_response.text})

    except chromadb.errors.ChromaError as e:
        logger.error(f"ChromaDB query error for question '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while searching for context.")
    except Exception as e:  # Catch-all for other errors (e.g., Gemini API issues)
        logger.error(f"Question answering failed for '{question}': {e}", exc_info=True)
        # Provide a generic error message to the client for security
        raise HTTPException(status_code=500, detail="Error generating answer.")
# To run this application (example):
# uvicorn main:app --reload
# (assuming this script is saved as main.py)