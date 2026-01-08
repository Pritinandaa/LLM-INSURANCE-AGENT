"""
Fireworks AI client for LLM inference and embeddings.
Provides wrapper around the Fireworks API with retry logic.
"""

import json
import os
import ssl
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time

from src.config import get_settings

# Check SSL settings BEFORE importing Fireworks
_settings = get_settings()
if not _settings.ssl_verify:
    # Disable SSL verification for corporate proxies/firewalls
    # Must be done BEFORE any HTTP client is initialized
    import warnings
    import urllib3
    import httpx

    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Set environment variables
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'

    # Monkey-patch ssl module
    ssl._create_default_https_context = ssl._create_unverified_context

    # Monkey-patch httpx to disable SSL verification by default
    _original_client_init = httpx.Client.__init__

    def _patched_client_init(self, *args, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_client_init(self, *args, **kwargs)

    httpx.Client.__init__ = _patched_client_init

    _original_async_client_init = httpx.AsyncClient.__init__

    def _patched_async_client_init(self, *args, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_async_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_async_client_init

# Now import Fireworks after SSL patching
from fireworks.client import Fireworks


class FireworksClient:
    """
    Wrapper for Fireworks AI API with structured output support.
    """

    def __init__(self):
        """Initialize the Fireworks client with API key."""
        settings = get_settings()

        # Initialize Fireworks client
        self.client = Fireworks(api_key=settings.fireworks_api_key)
        self.llm_model = settings.fireworks_llm_model
        self.embedding_model = settings.fireworks_embedding_model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60)
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60)
    )
    def generate_json(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output using the LLM.

        Args:
            prompt: User prompt
            schema: Optional Pydantic model for schema validation
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON as dictionary
        """
        messages = []

        # Build system prompt with JSON instruction
        full_system = "You must respond with valid JSON only. No additional text."
        if system_prompt:
            full_system = f"{system_prompt}\n\n{full_system}"
        messages.append({"role": "system", "content": full_system})

        # Add schema to prompt if provided
        if schema:
            schema_json = json.dumps(schema.model_json_schema(), indent=2)
            prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_json}"

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return json.loads(content)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60)
    )
    def generate_embeddings(
        self,
        texts: List[str],
        prefix: str = ""
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            prefix: Optional prefix to add to each text

        Returns:
            List of embedding vectors
        """
        if prefix:
            texts = [f"{prefix}{text}" for text in texts]

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def generate_embedding(self, text: str, prefix: str = "") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            prefix: Optional prefix

        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text], prefix)
        return embeddings[0]


@lru_cache()
def get_fireworks_client() -> FireworksClient:
    """Get cached Fireworks client instance."""
    return FireworksClient()
