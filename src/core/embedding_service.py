import os
import logging
from typing import List, Dict, Any
from functools import lru_cache
import vertexai
from vertexai.language_models import TextEmbeddingModel
from src.config import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using Vertex AI text-embedding-004.
    """

    def __init__(self):
        """Initialize Vertex AI and the embedding model."""
        try:
            settings = get_settings()
            # Initialize Vertex AI globally
            vertexai.init(project=settings.vertex_project_id, location=settings.vertex_location)
            self.model = TextEmbeddingModel.from_pretrained(settings.vertex_embedding_model or "text-embedding-004")
            logger.info(f"EmbeddingService initialized with model: {settings.vertex_embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingService: {e}")
            # Fallback for demo
            vertexai.init()
            self.model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    def embed_document(self, text: str) -> List[float]:
        """Generate embedding for a document."""
        try:
            embeddings = self.model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            return [0.0] * 768

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query."""
        try:
            embeddings = self.model.get_embeddings([text])
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        try:
            embeddings = self.model.get_embeddings(texts)
            return [e.values for e in embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return [[0.0] * 768 for _ in texts]

    def prepare_document_for_storage(
        self,
        document: Dict[str, Any],
        text_field: str = "content"
    ) -> Dict[str, Any]:
        """Prepare document for MongoDB with embedding."""
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
    return EmbeddingService()
