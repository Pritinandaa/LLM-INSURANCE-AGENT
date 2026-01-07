"""
Google Vertex AI client for LLM inference and embeddings.
Provides wrapper around the Vertex AI API with retry logic.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

from src.config import get_settings


class VertexAIClient:
    """
    Google Vertex AI client for text generation and embeddings.
    """

    def __init__(self):
        self.settings = get_settings()

        # Import Vertex AI libraries
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from vertexai.language_models import TextEmbeddingModel
            self.vertexai = vertexai
            self.GenerativeModel = GenerativeModel
            self.TextEmbeddingModel = TextEmbeddingModel
        except ImportError:
            raise ImportError("Vertex AI dependencies not installed. Install with: pip install google-cloud-aiplatform")

        # Initialize Vertex AI
        if self.settings.google_application_credentials:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.settings.google_application_credentials

        self.vertexai.init(
            project=self.settings.google_cloud_project,
            location=self.settings.google_cloud_region
        )

        # Initialize models
        self.llm_model = self.GenerativeModel("gemini-1.5-pro")
        self.embedding_model = self.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate text using Vertex AI Gemini model.
        """
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

            response = self.llm_model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )

            return response.text

        except Exception as e:
            raise Exception(f"Vertex AI text generation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.1, response_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate structured JSON output using Vertex AI Gemini model.
        """
        try:
            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nRespond with valid JSON only."

            # Configure generation for JSON
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": 2000,
            }

            if response_schema:
                generation_config["response_schema"] = response_schema
                generation_config["response_mime_type"] = "application/json"

            response = self.llm_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Parse JSON response
            result_text = response.text.strip()

            # Try to extract JSON if wrapped in markdown
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

            result_text = result_text.strip()

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text in a dict
                return {"response": result_text}

        except Exception as e:
            raise Exception(f"Vertex AI JSON generation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Vertex AI.
        """
        try:
            embeddings = []
            batch_size = 5  # Vertex AI has limits on batch size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])

            return embeddings

        except Exception as e:
            raise Exception(f"Vertex AI embedding generation failed: {str(e)}")

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return 768  # textembedding-gecko@003 produces 768-dimensional embeddings


# Global client instance
_vertex_client: Optional[VertexAIClient] = None


def get_vertex_client() -> VertexAIClient:
    """Get Vertex AI client singleton."""
    global _vertex_client
    if _vertex_client is None:
        _vertex_client = VertexAIClient()
    return _vertex_client</content>
<parameter name="filePath">c:\Users\rsnaraya\Downloads\underwriting-main (1)\underwriting-main\src\core\vertex_client.py