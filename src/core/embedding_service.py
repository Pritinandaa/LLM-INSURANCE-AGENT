"""
Embedding service for generating and managing vector embeddings.
Uses Google Vertex AI with text embedding models.
"""

from functools import lru_cache
from typing import List, Dict, Any

# Try Vertex AI first, fallback to Fireworks
try:
    from .vertex_client import get_vertex_client, VertexAIClient
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    from .fireworks_client import get_fireworks_client, FireworksClient


class EmbeddingService:
    """
    Service for generating embeddings for documents and queries.
    Uses Google Vertex AI text embeddings (768-dimensional).
    """

    # Embedding dimensions for Vertex AI textembedding-gecko
    EMBEDDING_DIMENSIONS = 768

    def __init__(self, client=None):
        """Initialize with Vertex AI or Fireworks client."""
        if VERTEX_AVAILABLE and not client:
            # Update to use specific env vars if available
            import os
            self.project_id = os.getenv("VERTEX_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
            self.location = os.getenv("VERTEX_LOCATION", os.getenv("GOOGLE_CLOUD_REGION", "us-central1"))
            
            # Re-init vertexai with specific project/location
            import vertexai
            vertexai.init(project=self.project_id, location=self.location)
            
            self.client = get_vertex_client()
            self.use_vertex = True
        else:
            self.client = client or get_fireworks_client()
            self.use_vertex = VERTEX_AVAILABLE and isinstance(self.client, VertexAIClient)

    def embed_document(self, text: str) -> List[float]:
        """
        Generate embedding for a document.
        Uses appropriate prefix for optimal retrieval performance.

        Args:
            text: Document text to embed

        Returns:
            768-dimensional embedding vector
        """
        if self.use_vertex:
            # Vertex AI embeddings
            return self.client.generate_embeddings([text])[0]
        else:
            # Fallback to Fireworks
            return self.client.generate_embedding(
                text,
                prefix="search_document: "
            )

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses appropriate prefix for optimal retrieval performance.

        Args:
            text: Query text to embed

        Returns:
            768-dimensional embedding vector
        """
        if self.use_vertex:
            # Vertex AI embeddings
            return self.client.generate_embeddings([text])[0]
        else:
            # Fallback to Fireworks
            return self.client.generate_embedding(
                text,
                prefix="search_query: "
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts

        Returns:
            List of embedding vectors
        """
        if self.use_vertex:
            # Vertex AI batch embeddings
            return self.client.generate_embeddings(texts)
        else:
            # Fallback to Fireworks
            return self.client.generate_embeddings(
                texts,
                prefix="search_document: "
            )

    def prepare_document_for_storage(
        self,
        document: Dict[str, Any],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """
        Prepare a document for MongoDB storage with embedding.

        Args:
            document: Document dictionary
            text_field: Field containing text to embed

        Returns:
            Document with 'embedding' field added
        """
        if text_field not in document:
            raise ValueError(f"Document missing required field: {text_field}")

        text = document[text_field]
        embedding = self.embed_document(text)

        return {
            **document,
            "embedding": embedding
        }


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance."""
    if VERTEX_AVAILABLE:
        client = get_vertex_client()
    else:
        client = get_fireworks_client()
    return EmbeddingService(client)
