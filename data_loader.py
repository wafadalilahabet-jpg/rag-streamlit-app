"""
PDF Embedding Pipeline
======================

This script:
1. Loads PDFs and extracts text.
2. Splits text into overlapping chunks for better context.
3. Embeds the chunks using OpenAI's embeddings API.
"""

# -------------------- Imports --------------------
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv


# -------------------- Environment --------------------
# Load API keys from .env
load_dotenv()


# -------------------- OpenAI Client --------------------
client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"  # Embedding model
EMBED_DIM = 3072                        # Output vector size


# -------------------- Text Splitter --------------------
# Overlapping chunks improve context retention
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


# -------------------- PDF Loader --------------------
def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF, extract its text, and split into chunks.

    Args:
        path: Path to the PDF file.

    Returns:
        List of text chunks.
    """
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


# -------------------- Embedding --------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert text chunks into embeddings.

    Args:
        texts: List of text chunks.

    Returns:
        List of embedding vectors (each of size EMBED_DIM).
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
