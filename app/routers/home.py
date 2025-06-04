import logging
import uuid
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import chromadb # For chromadb.errors.ChromaError

from app.core import processing, database, ai
# from app.config import GEMINI_MODEL_NAME # Not needed here directly if ai module handles it

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload-pdf", response_class=HTMLResponse)
async def handle_pdf_upload(file: UploadFile = File(...)):
    """
    Handles PDF file uploads, extracts text, chunks it, and stores embeddings in ChromaDB.
    """
    if not file.filename:
        logger.warning("Upload attempt with no filename.")
        raise HTTPException(status_code=400, detail="No file name provided.")
    if not file.filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    try:
        logger.info(f"Received PDF upload: {file.filename}")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            logger.warning(f"Uploaded PDF file is empty: {file.filename}")
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

        text_content = processing.extract_text_from_pdf(pdf_bytes)
        if not text_content.strip():
            logger.warning(f"No text could be extracted from PDF: {file.filename}")
            return HTMLResponse(
                content=f"<i class='fas fa-exclamation-triangle'></i> No text content found in <strong>{file.filename}</strong>. "
                        f"The file might be image-based, empty, or password-protected.",
                status_code=200 # Or 400 if this is considered a client error
            )

        text_chunks = processing.chunk_text(text_content)
        if not text_chunks:
             logger.warning(f"Text content from {file.filename} resulted in zero chunks.")
             return HTMLResponse(
                content=f"<i class='fas fa-info-circle'></i> Successfully processed <strong>{file.filename}</strong>, but no text chunks were generated (possibly very short content).",
                status_code=200
            )


        pdf_collection = database.get_collection()
        document_id = str(uuid.uuid4())
        chunk_ids = [f"{document_id}_{i}" for i, _ in enumerate(text_chunks)]
        chunk_metadatas = [{"source_document": file.filename}] * len(text_chunks)

        pdf_collection.add(
            documents=text_chunks, ids=chunk_ids, metadatas=chunk_metadatas
        )

        logger.info(
            f"Successfully stored {len(text_chunks)} chunks from {file.filename} "
            f"(Document ID: {document_id}) into ChromaDB."
        )
        return HTMLResponse(
            content=f"<i class='fas fa-check-circle'></i> Successfully uploaded and processed <strong>{file.filename}</strong>. "
            f"{len(text_chunks)} text chunks stored."
        )

    except RuntimeError as e: # Catches errors from processing.extract_text_from_pdf
        logger.error(f"PDF processing error during upload of {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error processing PDF: {str(e)}")
    except chromadb.errors.ChromaError as e:
        logger.error(f"ChromaDB error during upload of {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error during PDF processing.")
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF upload ({file.filename}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during file upload.")
    finally:
        await file.close()


@router.post("/ask", response_class=JSONResponse)
async def handle_ask_question(
    question: str = Form(...), k_results: int = Form(3)
) -> JSONResponse:
    """
    Receives a question, queries ChromaDB for relevant context,
    and uses Gemini AI to generate an answer.
    """
    if not question.strip():
        logger.warning("Received empty question.")
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not (1 <= k_results <= 10):
        logger.warning(f"Invalid k_results value: {k_results}")
        raise HTTPException(status_code=400, detail="k_results must be between 1 and 10.")

    try:
        logger.info(
            f"Received question: '{question}' (Querying for top {k_results} results)."
        )

        pdf_collection = database.get_collection()
        query_results = pdf_collection.query(
            query_texts=[question], n_results=k_results
        )

        if not query_results or not query_results.get("documents") or not query_results["documents"][0]:
            logger.warning(f"No relevant documents found in ChromaDB for question: '{question}'")
            return JSONResponse(
                content={"answer": "Sorry, I couldn't find any relevant information in the uploaded documents to answer your question."},
                status_code=200 # Or 404, but 200 with message is often better UX
            )

        context_parts = query_results["documents"][0]
        context = "\n\n---\n\n".join(context_parts)

        ai_answer = await ai.generate_answer_from_context(question, context)

        logger.info(f"Answer generated successfully for question: '{question}'")
        return JSONResponse(content={"answer": ai_answer})

    except chromadb.errors.ChromaError as e:
        logger.error(f"ChromaDB query error for question '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error while searching for context.")
    except RuntimeError as e: # E.g. if AI or DB not initialized
        logger.error(f"Runtime error during question answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Question answering failed for '{question}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating answer.")