"""
API routes for the underwriting system.
"""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse

from src import __version__
from src.config import get_settings
from src.core.mongodb_client import get_collection, Collections, get_mongodb_client
from src.pipeline.orchestrator import UnderwritingPipeline
from src.api.schemas import (
    QuoteRequest,
    QuoteResponse,
    QuoteDetailResponse,
    HealthResponse,
    ErrorResponse,
    PremiumSummaryResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Check the health status of the API and its dependencies.
    """
    settings = get_settings()

    # Check MongoDB connection
    mongodb_connected = False
    try:
        client = get_mongodb_client()
        client.admin.command('ping')
        mongodb_connected = True
    except Exception as e:
        logger.warning(f"MongoDB connection check failed: {e}")

    # Check Fireworks configuration
    fireworks_configured = bool(settings.fireworks_api_key and
                                settings.fireworks_api_key != "your_fireworks_api_key_here")

    return HealthResponse(
        status="healthy" if mongodb_connected and fireworks_configured else "degraded",
        version=__version__,
        mongodb_connected=mongodb_connected,
        fireworks_configured=fireworks_configured,
        timestamp=datetime.utcnow(),
    )


@router.post(
    "/api/quotes/process",
    response_model=QuoteResponse,
    tags=["Quotes"],
    summary="Process a quote request",
    description="Submit a broker email to generate an insurance quote using Microsoft AutoGen (Vertex AI)"
)
async def process_quote(request: QuoteRequest):
    """
    Process a broker email and generate an insurance quote.
    Uses Microsoft AutoGen multi-agent system powered by Vertex AI.
    """
    try:
        pipeline = UnderwritingPipeline()
        result = pipeline.process(request.email_content)

        if not result.success:
            return QuoteResponse(
                success=False,
                errors=result.errors,
                warnings=result.warnings,
            )

        # Build premium breakdown
        premium_breakdown = []
        if result.generated_quote:
            for item in result.generated_quote.premium_summary:
                premium_breakdown.append(PremiumSummaryResponse(
                    coverage_type=item.coverage_type,
                    premium=item.premium,
                    limits=item.limits,
                    deductible=item.deductible,
                ))

        return QuoteResponse(
            success=True,
            quote_id=result.quote_id,
            client_name=result.extracted_email.client_name if result.extracted_email else None,
            total_premium=result.modifier_result.adjusted_premium if result.modifier_result else None,
            premium_breakdown=premium_breakdown,
            risk_level=result.risk_assessment.overall_risk_level if result.risk_assessment else None,
            risk_score=result.risk_assessment.risk_score if result.risk_assessment else None,
            recommendation=result.risk_assessment.recommendation if result.risk_assessment else None,
            requires_approval=result.authority_check.requires_approval if result.authority_check else False,
            approval_reason=result.authority_check.approval_reason if result.authority_check else None,
            quote_letter=result.generated_quote.quote_letter if result.generated_quote else None,
            processing_time_seconds=result.metrics.total_duration_seconds if result.metrics else None,
            processing_mode="microsoft_autogen",
            warnings=result.warnings,
            errors=result.errors,
        )

    except Exception as e:
        logger.exception(f"Error processing quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/quotes/process-file",
    response_model=QuoteResponse,
    tags=["Quotes"],
    summary="Process a quote from file upload",
    description="Upload an email file (.txt or .eml) to generate a quote using Microsoft AutoGen"
)
async def process_quote_file(file: UploadFile = File(...)):
    """
    Process an uploaded email file and generate a quote.
    Accepts .txt or .eml files containing the broker email.
    """
    # Validate file type
    if not file.filename.endswith(('.txt', '.eml')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .txt or .eml file."
        )

    try:
        # Read file content
        content = await file.read()
        email_content = content.decode('utf-8')

        # Process through pipeline
        pipeline = UnderwritingPipeline()
        result = pipeline.process(email_content)

        if not result.success:
            return QuoteResponse(
                success=False,
                errors=result.errors,
                warnings=result.warnings,
            )

        # Build response (same as process_quote)
        premium_breakdown = []
        if result.generated_quote:
            for item in result.generated_quote.premium_summary:
                premium_breakdown.append(PremiumSummaryResponse(
                    coverage_type=item.coverage_type,
                    premium=item.premium,
                    limits=item.limits,
                    deductible=item.deductible,
                ))

        return QuoteResponse(
            success=True,
            quote_id=result.quote_id,
            client_name=result.extracted_email.client_name if result.extracted_email else None,
            total_premium=result.modifier_result.adjusted_premium if result.modifier_result else None,
            premium_breakdown=premium_breakdown,
            risk_level=result.risk_assessment.overall_risk_level if result.risk_assessment else None,
            risk_score=result.risk_assessment.risk_score if result.risk_assessment else None,
            recommendation=result.risk_assessment.recommendation if result.risk_assessment else None,
            requires_approval=result.authority_check.requires_approval if result.authority_check else False,
            approval_reason=result.authority_check.approval_reason if result.authority_check else None,
            quote_letter=result.generated_quote.quote_letter if result.generated_quote else None,
            processing_time_seconds=result.metrics.total_duration_seconds if result.metrics else None,
            processing_mode="microsoft_autogen",
            warnings=result.warnings,
            errors=result.errors,
        )

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode file. Please ensure it's a valid text file."
        )
    except Exception as e:
        logger.exception(f"Error processing quote file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/quotes/{quote_id}",
    response_model=QuoteDetailResponse,
    tags=["Quotes"],
    summary="Get quote details",
    description="Retrieve a previously generated quote by ID"
)
async def get_quote(quote_id: str):
    """
    Retrieve a quote by its ID.
    Returns full quote details including all underwriting data.
    """
    try:
        quotes_collection = get_collection(Collections.QUOTES)
        quote_doc = quotes_collection.find_one({"_id": quote_id})

        if not quote_doc:
            raise HTTPException(status_code=404, detail=f"Quote {quote_id} not found")

        # Extract data from stored document
        extracted = quote_doc.get("extracted_email", {})
        industry = quote_doc.get("industry_classification", {})
        premium = quote_doc.get("premium_calculation", {})
        modifier = quote_doc.get("modifier_result", {})
        risk = quote_doc.get("risk_assessment", {})
        authority = quote_doc.get("authority_check", {})
        coverage = quote_doc.get("coverage_analysis", {})
        generated = quote_doc.get("generated_quote", {})
        metrics = quote_doc.get("metrics", {})

        # Build premium breakdown
        premium_breakdown = []
        for item in generated.get("premium_summary", []):
            premium_breakdown.append(PremiumSummaryResponse(
                coverage_type=item.get("coverage_type", ""),
                premium=item.get("premium", 0),
                limits=item.get("limits"),
                deductible=item.get("deductible"),
            ))

        return QuoteDetailResponse(
            success=True,
            quote_id=quote_id,
            generated_at=quote_doc.get("saved_at", datetime.utcnow()),
            client_name=extracted.get("client_name", "Unknown"),
            industry=industry.get("industry_name"),
            bic_code=industry.get("bic_code"),
            total_premium=modifier.get("adjusted_premium", 0),
            base_premium=premium.get("total_base_premium"),
            modifier_adjustment=modifier.get("total_modifier_impact"),
            premium_breakdown=premium_breakdown,
            risk_level=risk.get("overall_risk_level"),
            risk_score=risk.get("risk_score"),
            risk_factors=risk.get("risk_factors", []),
            positive_factors=risk.get("positive_factors", []),
            authority_level=authority.get("authority_level"),
            requires_approval=authority.get("requires_approval", False),
            approval_reason=authority.get("approval_reason"),
            recommended_endorsements=coverage.get("recommended_endorsements", []),
            coverage_gaps=coverage.get("coverage_gaps", []),
            quote_letter=generated.get("quote_letter"),
            terms_and_conditions=generated.get("terms_and_conditions", []),
            processing_time_seconds=metrics.get("total_duration_seconds"),
            warnings=quote_doc.get("warnings", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/quotes",
    tags=["Quotes"],
    summary="List recent quotes",
    description="Get a list of recently processed quotes"
)
async def list_quotes(limit: int = 20, skip: int = 0):
    """
    List recent quotes with pagination.
    """
    try:
        quotes_collection = get_collection(Collections.QUOTES)

        quotes = list(quotes_collection.find(
            {},
            {
                "_id": 1,
                "extracted_email.client_name": 1,
                "modifier_result.adjusted_premium": 1,
                "risk_assessment.overall_risk_level": 1,
                "saved_at": 1,
            }
        ).sort("saved_at", -1).skip(skip).limit(limit))

        return {
            "quotes": [
                {
                    "quote_id": q["_id"],
                    "client_name": q.get("extracted_email", {}).get("client_name", "Unknown"),
                    "total_premium": q.get("modifier_result", {}).get("adjusted_premium"),
                    "risk_level": q.get("risk_assessment", {}).get("overall_risk_level"),
                    "created_at": q.get("saved_at"),
                }
                for q in quotes
            ],
            "count": len(quotes),
            "skip": skip,
            "limit": limit,
        }

    except Exception as e:
        logger.exception(f"Error listing quotes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
