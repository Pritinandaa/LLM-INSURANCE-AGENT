"""
Vector search service for semantic document retrieval.
Implements RAG (Retrieval-Augmented Generation) pattern.
"""

from functools import lru_cache
from typing import List, Dict, Any, Optional
from pymongo.collection import Collection

from .mongodb_client import get_collection
from .embedding_service import get_embedding_service, EmbeddingService


class VectorSearchService:
    """
    Service for semantic search using MongoDB Atlas Vector Search.
    Supports RAG pattern for grounding LLM responses in documents.
    """

    def __init__(self, embedding_service: EmbeddingService):
        """Initialize with embedding service."""
        self.embedding_service = embedding_service

    def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        min_score: float = 0.5,
        filter_query: Optional[Dict[str, Any]] = None,
        index_name: str = "vector_index",
        vector_field: str = "embedding",
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            collection_name: MongoDB collection to search
            query: Search query text
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            filter_query: Optional MongoDB filter
            index_name: Name of the vector search index
            vector_field: Field containing embeddings

        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)

        # Get collection
        collection = get_collection(collection_name)

        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "queryVector": query_embedding,
                    "path": vector_field,
                    "numCandidates": limit * 10,  # Oversample for better recall
                    "limit": limit,
                    "index": index_name,
                }
            },
            {
                "$addFields": {
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        # Add filter if provided
        if filter_query:
            pipeline[0]["$vectorSearch"]["filter"] = filter_query

        # Add score threshold filter
        pipeline.append({
            "$match": {
                "score": {"$gte": min_score}
            }
        })

        # Execute search
        results = list(collection.aggregate(pipeline))

        return results

    def search_with_context(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        context_fields: Optional[List[str]] = None,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Search and return results with formatted context for LLM.

        Args:
            collection_name: Collection to search
            query: Search query
            limit: Number of results
            context_fields: Fields to include in context

        Returns:
            Tuple of (results, formatted_context_string)
        """
        results = self.search(collection_name, query, limit)

        # Format context for LLM consumption
        context_parts = []
        for i, doc in enumerate(results, 1):
            if context_fields:
                doc_text = "\n".join(
                    f"{field}: {doc.get(field, 'N/A')}"
                    for field in context_fields
                )
            else:
                # Default: use 'content' and 'name' fields
                name = doc.get("name", doc.get("title", f"Document {i}"))
                content = doc.get("content", str(doc))
                doc_text = f"[{name}]\n{content}"

            context_parts.append(f"--- Document {i} (Score: {doc.get('score', 0):.2f}) ---\n{doc_text}")

        context_string = "\n\n".join(context_parts)

        return results, context_string

    def rag_query(
        self,
        collection_name: str,
        query: str,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """
        Perform RAG query: retrieve relevant documents and format for LLM.

        Args:
            collection_name: Collection to search
            query: User query
            limit: Number of documents to retrieve

        Returns:
            Dict with 'documents', 'context', and 'query'
        """
        results, context = self.search_with_context(
            collection_name, query, limit
        )

        return {
            "query": query,
            "documents": results,
            "context": context,
            "document_count": len(results),
        }


@lru_cache()
def get_vector_search_service() -> VectorSearchService:
    """Get cached vector search service instance."""
    embedding_service = get_embedding_service()
    return VectorSearchService(embedding_service)
