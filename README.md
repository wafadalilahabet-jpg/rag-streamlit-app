# Streamlit RAG App

## Overview
This application provides a Streamlit interface for interacting with a Retrieval-Augmented Generation (RAG) pipeline.

The app allows users to:
- Upload PDF documents for ingestion into a vector database.
- Trigger ingestion events handled by Inngest.
- Ask natural language questions against the ingested PDFs.
- View generated answers along with their sources.

## Features
- **Custom Theme**: Teal, honey, lilac, and cream theme with styled buttons and inputs.
- **PDF Upload**: Upload PDFs which are then saved locally and ingested into the RAG pipeline.
- **Query Interface**: Ask questions and retrieve answers supported by document chunks.
- **Inngest Integration**: Event-driven architecture for ingestion and query handling.

## Project Structure
```
.
├── app.py              # Main Streamlit application
├── schemas.py          # Pydantic models for data structures
├── helpers.py          # Utility functions for Inngest and PDF handling
├── uploads/            # Directory for uploaded PDFs
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone this repository:
```bash

git clone https://github.com/wafadalilahabet-jpg/rag-streamlit-app.git

cd rag-streamlit-app
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate    # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables in a `.env` file:
```env
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open the app in your browser at `http://localhost:8501`.

3. Upload a PDF and trigger ingestion.

4. Use the query section to ask questions about your uploaded PDFs.

## Requirements
- Python 3.10+
- Streamlit
- Inngest Python SDK
- Requests
- Pydantic
- python-dotenv

## License
This project is licensed under the MIT License.

