"""
FastAPI application entry point.
AI-Powered Insurance Underwriting System
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.config import get_settings
from src.api.routes import router
from src.core.mongodb_client import get_mongodb_client, close_mongodb_client


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Insurance Underwriting API...")
    settings = get_settings()
    logger.info(f"API Version: {__version__}")
    logger.info(f"Environment: {'Production' if settings.api_host != 'localhost' else 'Development'}")

    # Test MongoDB connection
    try:
        client = get_mongodb_client()
        client.admin.command('ping')
        logger.info("MongoDB connection established")
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        logger.warning("API will start but database operations will fail")

    yield

    # Shutdown
    logger.info("Shutting down Insurance Underwriting API...")
    close_mongodb_client()
    logger.info("Cleanup complete")


# Create FastAPI application
app = FastAPI(
    title="Insurance Underwriting API",
    description="""
    AI-Powered Insurance Underwriting System

    This API provides automated insurance quote generation using:
    - **Fireworks AI** (Llama 3.3 70B) for natural language processing
    - **MongoDB Atlas** for document storage and vector search
    - **RAG** (Retrieval-Augmented Generation) for accurate, grounded decisions

    ## Features

    - Parse unstructured broker emails
    - Classify business industries automatically
    - Calculate premiums based on rating manuals
    - Apply risk-based modifiers
    - Generate professional quote letters

    ## Quick Start

    1. POST an email to `/api/quotes/process`
    2. Receive a complete insurance quote in seconds
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Insurance Underwriting API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
