"""
Qdrant Setup
============

This script:
1. Connects to a Qdrant instance (Cloud or Docker).
2. Loads environment variables for secure connection.
3. Provides a QdrantStorage class to:
   - Create collections if needed.
   - Upsert (insert or update) vectors with metadata.
   - Search for similar vectors and retrieve contexts/sources.
"""

# -------------------- Imports --------------------
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# -------------------- Environment --------------------
# Load API keys from .env
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# -------------------- Connection Test --------------------
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections()
    print(f"Connected to Qdrant. Collections: {collections.collections}")
except Exception as e:
    print(f"Connection failed: {e}")


# -------------------- Qdrant Storage --------------------
class QdrantStorage:
    """
    Storage handler for Qdrant Cloud.

    Responsibilities:
    1. Connect to Qdrant.
    2. Ensure the target collection exists.
    3. Upsert vectors with payloads (metadata).
    4. Search for similar vectors.
    """

    def __init__(self, collection: str = "docs", dim: int = 3072):
        """
        Initialize the Qdrant storage.

        Args:
            collection: Name of the Qdrant collection.
            dim: Dimension of vectors (must match embedding size).
        """
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            print(f"Collection '{self.collection}' not found. Creating new collection...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection}' created successfully.")
        else:
            print(f"Collection '{self.collection}' already exists.")

    def upsert(self, ids, vectors, payloads):
        """
        Insert or update vectors in the collection.

        Args:
            ids: List of unique IDs for vectors.
            vectors: List of embedding vectors.
            payloads: List of metadata dicts associated with each vector.
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        print(f"Upserted {len(points)} points into '{self.collection}'.")

    def search(self, query_vector, top_k: int = 5):
        """
        Search for similar vectors in the collection.

        Args:
            query_vector: Embedding vector to query.
            top_k: Number of results to return.

        Returns:
            dict: Contains 'contexts' (texts) and 'sources' (metadata).
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
        )

        contexts = []
        sources = set()
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        print(f"Found {len(contexts)} matches.")
        return {"contexts": contexts, "sources": list(sources)}
