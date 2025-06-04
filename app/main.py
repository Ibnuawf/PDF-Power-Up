import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # Optional: if you need CORS

from app.config import GEMINI_API_KEY # To check if AI should be initialized
from app.core import database, ai
from app.routers import home, qa

# ===== Logger Setup =====
# Basic config should be done once, as early as possible.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup: Initializing core services...")
    try:
        database.initialize_db()
        if GEMINI_API_KEY: # Only initialize AI if key is present
            ai.initialize_ai()
        else:
            logger.warning("GEMINI_API_KEY not found. AI features will be disabled.")
        logger.info("Core services initialization complete.")
    except Exception as e:
        logger.critical(f"Fatal error during core service initialization: {e}", exc_info=True)
        # You might want the app to not start or to enter a degraded mode
        # For now, we'll let it start, but AI/DB might not work.
        # Consider raising the error to prevent startup: raise e
    yield
    # Shutdown
    logger.info("Application shutdown: Closing core services...")
    ai.close_ai()
    database.close_db()
    logger.info("Core services closed.")


app = FastAPI(
    title="PDF QA Assistant Pro",
    description="Upload PDFs and ask questions using generative AI.",
    version="1.0.0",
    lifespan=lifespan,
)

# Optional: Add CORS middleware if your frontend is on a different domain
# app.add_middleware(
# CORSMiddleware,
# allow_origins=["*"], # Or specify your frontend origin
# allow_credentials=True,
# allow_methods=["*"],
# allow_headers=["*"],
# )

# Include routers
app.include_router(home.router, tags=["Home"])
app.include_router(qa.router, prefix="/api/v1", tags=["QA"]) # Added a prefix for API routes

# Generic Exception Handler (optional, but good for consistent error responses)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )

logger.info("FastAPI application configured and ready.")