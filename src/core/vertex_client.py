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
        self.mock_mode = False

        # Import Vertex AI libraries
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from vertexai.language_models import TextEmbeddingModel
            self.vertexai = vertexai
            self.GenerativeModel = GenerativeModel
            self.TextEmbeddingModel = TextEmbeddingModel
        except ImportError:
            self.mock_mode = True
            print("⚠️  Vertex AI dependencies not installed. Using mock mode.")
            return

        # Initialize Vertex AI
        try:
            if self.settings.google_application_credentials:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.settings.google_application_credentials

            self.vertexai.init(
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_region
            )

            # Initialize models only if not in mock mode
            if not self.mock_mode:
                self.llm_model = self.GenerativeModel("gemini-pro")
                self.embedding_model = self.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

                # Test model access to ensure it works
                try:
                    test_response = self.llm_model.generate_content("test", generation_config={"max_output_tokens": 1})
                    print("✅ Vertex AI models initialized successfully")
                except Exception as e:
                    print(f"⚠️  Vertex AI model access failed: {e}. Using mock mode.")
                    self.mock_mode = True
        except Exception as e:
            print(f"⚠️  Vertex AI initialization failed: {e}. Using mock mode.")
            self.mock_mode = True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate text using Vertex AI Gemini model.
        """
        if self.mock_mode:
            return f"Mock response for: {prompt[:100]}..."

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

        except AttributeError:
            # Model not initialized (mock mode)
            return f"Mock response for: {prompt[:100]}..."
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
        print(f"DEBUG: mock_mode = {self.mock_mode}")  # Debug output

        if self.mock_mode:
            # Return mock data based on the schema or prompt content
            if response_schema and "underwriting_decision" in response_schema.get("properties", {}):
                return {
                    "underwriting_decision": "approved",
                    "premium": 2500.00,
                    "coverage_details": "Standard commercial liability coverage",
                    "risk_factors": ["Industry risk: Technology startup"],
                    "recommendations": ["Annual review recommended"]
                }
            elif response_schema and "industry" in response_schema.get("properties", {}):
                return {
                    "industry": "Technology",
                    "confidence": 0.95,
                    "sub_industry": "Software Development"
                }
            elif response_schema and "authority_check" in response_schema.get("properties", {}):
                return {
                    "has_authority": True,
                    "authority_level": "Full underwriting authority",
                    "limits": {"min": 1000, "max": 100000}
                }
            elif "industry" in prompt.lower():
                return {
                    "industry": "Technology",
                    "confidence": 0.95,
                    "sub_industry": "Software Development"
                }
            elif "authority" in prompt.lower():
                return {
                    "has_authority": True,
                    "authority_level": "Full underwriting authority",
                    "limits": {"min": 1000, "max": 100000}
                }
            else:
                return {"mock": True, "message": "Mock JSON response generated"}

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

        except AttributeError:
            # Model not initialized (mock mode) - mock responses already handled above
            pass
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
        if self.mock_mode:
            # Return mock embeddings (768-dimensional vectors)
            import random
            mock_embeddings = []
            for _ in texts:
                # Generate a mock 768-dimensional embedding vector
                embedding = [random.uniform(-1.0, 1.0) for _ in range(768)]
                # Normalize to unit vector
                norm = sum(x**2 for x in embedding) ** 0.5
                embedding = [x / norm for x in embedding]
                mock_embeddings.append(embedding)
            return mock_embeddings

        try:
            embeddings = []
            batch_size = 5  # Vertex AI has limits on batch size

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])

            return embeddings

        except AttributeError:
            # Model not initialized (mock mode) - mock embeddings already handled above
            pass
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
    return _vertex_client