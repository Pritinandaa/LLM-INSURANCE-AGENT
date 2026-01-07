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


_client = None


class MockCollection:
    def __init__(self, name):
        self.name = name
    def find_one(self, *args, **kwargs): return None
    def find(self, *args, **kwargs): return self
    def sort(self, *args, **kwargs): return self
    def skip(self, *args, **kwargs): return self
    def limit(self, *args, **kwargs): return []
    def insert_one(self, *args, **kwargs): return None
    def replace_one(self, *args, **kwargs): return None
    def update_one(self, *args, **kwargs): return None
    def delete_one(self, *args, **kwargs): return None
    def count_documents(self, *args, **kwargs): return 0
    def __iter__(self): return iter([])

class MockDatabase:
    def __getitem__(self, name):
        return MockCollection(name)

class MockMongoClient:
    def __init__(self, *args, **kwargs):
        self.admin = self
    def __getitem__(self, name):
        return MockDatabase()
    def command(self, *args, **kwargs): return {"ok": 1}
    def close(self): pass

def get_mongodb_client() -> MongoClient:
    """
    Get MongoDB client singleton.
    Fallback to Mock if connection fails.
    """
    global _client
    if _client is None:
        settings = get_settings()
        try:
            # Short timeout to fail fast
            client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=2000)
            client.admin.command('ping')
            _client = client
        except Exception as e:
            # Fallback to mock
            print(f"⚠️ MongoDB connection failed ({e}). Using MockDB.")
            _client = MockMongoClient()
    return _client

def get_database():
    """Get the configured database."""
    client = get_mongodb_client()
    # Handle mock client
    if isinstance(client, MockMongoClient):
        return client["mock_db"]
    settings = get_settings()
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
