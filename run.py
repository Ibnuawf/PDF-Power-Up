import uvicorn
import logging

if __name__ == "__main__":
    logging.info("Starting PDF QA Assistant Pro server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)