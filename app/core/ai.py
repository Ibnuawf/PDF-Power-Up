import logging
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level variable to hold the generative model
generative_model = None

def initialize_ai():
    """Initializes the Gemini AI model."""
    global generative_model
    try:
        if not GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set. AI functionalities will be disabled.")
            # Optionally raise an error if AI is critical
            # raise ValueError("GEMINI_API_KEY is not set, cannot initialize AI.")
            return # Allow app to run without AI if key is missing

        genai.configure(api_key=GEMINI_API_KEY)
        generative_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Gemini AI configured successfully with model: '{GEMINI_MODEL_NAME}'.")
    except Exception as e:
        logger.critical(f"Failed to initialize Gemini AI: {e}", exc_info=True)
        # Optionally re-raise to halt app startup if AI is critical
        raise RuntimeError(f"Gemini AI initialization failed: {e}") from e

async def generate_answer_from_context(question: str, context: str) -> str:
    """Generates an answer using the AI model based on provided context and question."""
    if generative_model is None:
        logger.warning("AI model not initialized. Cannot generate answer.")
        return "AI model is not available. Please check configuration."

    prompt = (
        "Using ONLY the information provided in the following context (from uploaded PDF documents), "
        "answer the user's question accurately, clearly, and respectfully. "
        "If the answer is not present in the context, respond with: "
        "'Sorry, I could not find relevant information in the provided documents.'\n\n"
        "Context (from PDF documents):\n"
        f"{context}\n\n"
        "User's Question:\n"
        f"{question}\n\n"
        "Your Answer (cite the context if possible):"
    )
    logger.debug(f"Prompt sent to Gemini: {prompt[:500]}...")

    try:
        ai_response = await generative_model.generate_content_async(prompt)
        return ai_response.text
    except Exception as e:
        logger.error(f"Error during Gemini content generation: {e}", exc_info=True)
        return "An error occurred while trying to generate an answer."

def close_ai():
    """Placeholder for any AI model cleanup if needed."""
    global generative_model
    generative_model = None
    logger.info("Gemini AI resources released.")