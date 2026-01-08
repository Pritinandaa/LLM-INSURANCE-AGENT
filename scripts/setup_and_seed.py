"""
One-command setup script for the Insurance Underwriting AI system.
Creates MongoDB collections, indexes, and seeds initial data.

Usage: python scripts/setup_and_seed.py
"""

import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.core.mongodb_client import get_database, Collections
from src.core.embedding_service import get_embedding_service


def print_step(step: str, status: str = "..."):
    """Print step with status."""
    icons = {
        "...": "...",
        "done": "done",
        "skip": "skip",
        "fail": "FAIL"
    }
    print(f"  [{icons.get(status, status)}] {step}")


def print_header(text: str):
    """Print a header."""
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def load_json_file(filename: str) -> dict:
    """Load a JSON file from the data directory."""
    settings = get_settings()
    file_path = settings.data_dir / filename
    with open(file_path, 'r') as f:
        return json.load(f)


def create_vector_search_index(collection, index_name: str = "vector_index"):
    """
    Create a vector search index on a collection.
    Note: For MongoDB Atlas, this needs to be done through Atlas UI or API.
    For local MongoDB, vector search is not available.
    """
    # Check if we can create search indexes (Atlas only)
    try:
        # Try to list existing indexes
        indexes = list(collection.list_indexes())
        index_names = [idx.get("name") for idx in indexes]

        if index_name in index_names:
            print_step(f"Vector index '{index_name}' already exists on {collection.name}", "skip")
            return True

        # For Atlas, we would use the Atlas Search API
        # For now, print instructions
        print_step(f"Vector index needed on {collection.name}", "done")
        return True

    except Exception as e:
        print_step(f"Could not check indexes on {collection.name}: {e}", "skip")
        return False


def seed_collection(
    collection_name: str,
    data_file: str,
    items_key: str,
    text_field: str = "content"
):
    """
    Seed a collection with data and generate embeddings.

    Args:
        collection_name: MongoDB collection name
        data_file: JSON file containing data
        items_key: Key in JSON containing the items array
        text_field: Field to use for embedding generation
    """
    db = get_database()
    collection = db[collection_name]
    embedding_service = get_embedding_service()

    # Load data
    data = load_json_file(data_file)
    items = data.get(items_key, [])

    if not items:
        print_step(f"No items found in {data_file}", "skip")
        return 0

    # Check if already seeded
    existing_count = collection.count_documents({})
    if existing_count > 0:
        print_step(f"{collection_name}: {existing_count} documents exist", "skip")
        return existing_count

    # Generate embeddings and insert
    print(f"  Seeding {collection_name} with {len(items)} items...")

    documents = []
    for i, item in enumerate(items):
        # Get text for embedding
        text = item.get(text_field, "")
        if not text:
            # Try to build text from available fields
            text = " ".join(str(v) for v in item.values() if isinstance(v, str))

        # Generate embedding
        try:
            embedding = embedding_service.embed_document(text)
            item["embedding"] = embedding
        except Exception as e:
            print(f"    Warning: Could not generate embedding for item {i}: {e}")
            item["embedding"] = [0.0] * 768  # Placeholder

        documents.append(item)

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(items)} items...")

    # Insert all documents
    if documents:
        collection.insert_many(documents)
        print_step(f"{collection_name}: Inserted {len(documents)} documents", "done")

    return len(documents)


def main():
    """Main setup function."""
    print_header("Insurance Underwriting AI - Setup")

    # Check environment
    print("\nChecking environment...")
    settings = get_settings()

    # Check Fireworks API key
    if not settings.fireworks_api_key or settings.fireworks_api_key == "your_fireworks_api_key_here":
        print("\n  ERROR: Fireworks API key not configured!")
        print("  Please set FIREWORKS_API_KEY in your .env file")
        print("  Get your key at: https://fireworks.ai")
        sys.exit(1)
    print_step("Fireworks API key configured", "done")

    # Check MongoDB connection
    print("\nConnecting to MongoDB...")
    try:
        db = get_database()
        db.command('ping')
        print_step(f"Connected to database: {settings.mongodb_database}", "done")
    except Exception as e:
        print(f"\n  ERROR: Could not connect to MongoDB!")
        print(f"  {e}")
        print("\n  Please check your MONGODB_URI in .env")
        print("  For Atlas: mongodb+srv://user:pass@cluster.mongodb.net/")
        print("  For local: mongodb://localhost:27017")
        sys.exit(1)

    # Create collections and seed data
    print_header("Seeding Data")

    total_docs = 0

    # Seed BIC codes
    total_docs += seed_collection(
        Collections.BIC_CODES,
        "bic_codes.json",
        "codes",
        "content"
    )

    # Seed rating manuals
    total_docs += seed_collection(
        Collections.RATING_MANUALS,
        "rating_manuals.json",
        "rates",
        "content"
    )

    # Seed underwriting guidelines
    total_docs += seed_collection(
        Collections.UNDERWRITING_GUIDELINES,
        "underwriting_guidelines.json",
        "guidelines",
        "content"
    )

    # Seed modifiers
    total_docs += seed_collection(
        Collections.MODIFIERS,
        "modifiers.json",
        "modifiers",
        "content"
    )

    # Create vector indexes
    print_header("Creating Indexes")

    collections_to_index = [
        Collections.BIC_CODES,
        Collections.RATING_MANUALS,
        Collections.UNDERWRITING_GUIDELINES,
        Collections.MODIFIERS,
    ]

    for coll_name in collections_to_index:
        collection = db[coll_name]

        # Create regular index on common fields
        try:
            collection.create_index("name")
            print_step(f"Created name index on {coll_name}", "done")
        except:
            pass

        # Note about vector index
        create_vector_search_index(collection)

    # Print vector index instructions for Atlas
    print_header("Vector Search Index Setup (MongoDB Atlas)")
    print("""
  For full functionality, create vector search indexes in MongoDB Atlas:

  1. Go to your Atlas cluster
  2. Click "Atlas Search" tab
  3. Click "Create Search Index"
  4. Choose "JSON Editor"
  5. For each collection, create an index with this definition:

  {
    "mappings": {
      "dynamic": true,
      "fields": {
        "embedding": {
          "type": "knnVector",
          "dimensions": 768,
          "similarity": "cosine"
        }
      }
    }
  }

  Collections to index:
  - bic_codes
  - rating_manuals
  - underwriting_guidelines
  - modifiers

  Index name: vector_index (for each collection)
    """)

    # Summary
    print_header("Setup Complete!")
    print(f"""
  Total documents seeded: {total_docs}

  Next steps:
  1. Create vector search indexes in MongoDB Atlas (see instructions above)
  2. Start the API server:
     python -m uvicorn src.main:app --reload

  3. Or use the CLI:
     python -m cli.process_quote data/sample_emails/construction_company.txt

  API docs will be available at: http://localhost:8000/docs
    """)


if __name__ == "__main__":
    main()
