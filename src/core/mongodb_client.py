"""
MongoDB client singleton for database operations.
Provides connection management and collection access.
"""

from functools import lru_cache
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from src.config import get_settings


_client: Optional[MongoClient] = None


def get_mongodb_client() -> MongoClient:
    """
    Get MongoDB client singleton.
    Creates a new connection if one doesn't exist.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = MongoClient(settings.mongodb_uri)
    return _client


def get_database() -> Database:
    """Get the configured database."""
    settings = get_settings()
    client = get_mongodb_client()
    return client[settings.mongodb_database]


def get_collection(collection_name: str) -> Collection:
    """Get a specific collection from the database."""
    db = get_database()
    return db[collection_name]


def close_mongodb_client() -> None:
    """Close the MongoDB client connection."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


# Collection names as constants
class Collections:
    """MongoDB collection names."""
    BIC_CODES = "bic_codes"
    RATING_MANUALS = "rating_manuals"
    UNDERWRITING_GUIDELINES = "underwriting_guidelines"
    MODIFIERS = "modifiers"
    QUOTES = "quotes"
    PROCESSED_EMAILS = "processed_emails"
