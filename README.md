# PDF Power-Up: PDF QA Assistant Pro

PDF Power-Up is a FastAPI-based web application that allows users to upload PDF documents and ask questions about their content using advanced generative AI (Google Gemini) and vector search (ChromaDB). The app extracts, chunks, and embeds PDF text, enabling semantic search and context-aware answers to user queries.

## Features

- **PDF Upload:** Securely upload PDF files for processing.
- **Text Extraction & Chunking:** Extracts text from PDFs and splits it into manageable chunks.
- **Vector Embeddings:** Uses Sentence Transformers to embed text for semantic search.
- **ChromaDB Integration:** Stores and retrieves document chunks for efficient context retrieval.
- **AI-Powered Q&A:** Leverages Google Gemini to answer questions based on PDF content.
- **Modern UI:** Clean, responsive interface for uploading, querying, and viewing answers.

## Getting Started

### Prerequisites

- Python 3.10+
- [Google Gemini API Key](https://ai.google.dev/)
- (Optional) [ChromaDB](https://www.trychroma.com/) for persistent vector storage

### Installation

1. **Clone the repository:**
   ```powershell
   git clone <your-repo-url>
   cd p1
   ```
2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

### Running the App

```powershell
python run.py
```

- The app will be available at [http://localhost:8000](http://localhost:8000)

## Usage

1. Open the app in your browser.
2. Upload a PDF document.
3. Ask questions about the uploaded PDF.
4. View AI-generated answers based on the document content.

## Project Structure

```
app/
  main.py           # FastAPI app entry point
  config.py         # Configuration and environment variables
  core/             # Core logic: AI, database, processing
  routers/          # API route definitions (home, qa)
  templates/        # Jinja2 HTML templates
requirements.txt    # Python dependencies
run.py              # App runner script
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Google Gemini](https://ai.google.dev/)
- [Sentence Transformers](https://www.sbert.net/)
