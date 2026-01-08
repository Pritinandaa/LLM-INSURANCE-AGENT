"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QuoteRequest(BaseModel):
    """Request body for submitting a quote."""
    email_content: str = Field(
        ...,
        description="The raw email content from the broker",
        min_length=50,
        examples=[
            "Subject: Quote Request\n\nHi, I need a quote for ABC Corp, a construction company..."
        ]
    )


class QuoteRequestFile(BaseModel):
    """Response after processing file upload."""
    filename: str
    email_content: str


class PremiumSummaryResponse(BaseModel):
    """Premium summary in quote response."""
    coverage_type: str
    premium: float
    limits: Optional[str] = None
    deductible: Optional[float] = None


class QuoteResponse(BaseModel):
    """Response containing the generated quote."""
    success: bool
    quote_id: Optional[str] = None
    client_name: Optional[str] = None
    total_premium: Optional[float] = None
    premium_breakdown: List[PremiumSummaryResponse] = Field(default_factory=list)
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    recommendation: Optional[str] = None
    requires_approval: bool = False
    approval_reason: Optional[str] = None
    quote_letter: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "quote_id": "Q-20240301-ABC12345",
                "client_name": "ABC Construction Corp",
                "total_premium": 165375.00,
                "premium_breakdown": [
                    {"coverage_type": "general_liability", "premium": 127500.00, "limits": "$2M/$4M"},
                    {"coverage_type": "auto_liability", "premium": 18750.00, "limits": "$1M CSL"}
                ],
                "risk_level": "MEDIUM",
                "risk_score": 45.0,
                "recommendation": "ACCEPT",
                "requires_approval": False,
                "processing_time_seconds": 47.3
            }
        }


class QuoteDetailResponse(BaseModel):
    """Detailed quote response including all pipeline data."""
    success: bool
    quote_id: str
    generated_at: datetime

    # Client info
    client_name: str
    industry: Optional[str] = None
    bic_code: Optional[str] = None

    # Premium details
    total_premium: float
    base_premium: Optional[float] = None
    modifier_adjustment: Optional[float] = None
    premium_breakdown: List[PremiumSummaryResponse] = Field(default_factory=list)

    # Risk assessment
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    positive_factors: List[str] = Field(default_factory=list)

    # Authority
    authority_level: Optional[str] = None
    requires_approval: bool = False
    approval_reason: Optional[str] = None

    # Coverage
    recommended_endorsements: List[Dict[str, Any]] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)

    # Quote document
    quote_letter: Optional[str] = None
    terms_and_conditions: List[str] = Field(default_factory=list)

    # Metrics
    processing_time_seconds: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str
    mongodb_connected: bool
    fireworks_configured: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
