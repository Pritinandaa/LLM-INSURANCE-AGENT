"""
Pydantic models for the underwriting pipeline.
These models define the data structures passed between pipeline steps.
"""

from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CoverageRequest(BaseModel):
    """Coverage type and limits requested."""
    coverage_type: str = Field(description="Type of coverage (e.g., general_liability, auto, property)")
    limits: Optional[str] = Field(default=None, description="Coverage limits if specified")
    additional_details: Optional[str] = Field(default=None, description="Any additional coverage details")


class BrokerContact(BaseModel):
    """Broker contact information."""
    name: Optional[str] = Field(default=None, description="Broker name")
    email: Optional[str] = Field(default=None, description="Broker email")
    phone: Optional[str] = Field(default=None, description="Broker phone")
    brokerage: Optional[str] = Field(default=None, description="Brokerage company name")


class ExtractedEmail(BaseModel):
    """Step 1: Extracted information from broker email."""
    client_name: str = Field(description="Name of the client/company")
    industry_description: str = Field(description="Description of the business/industry")
    location: Optional[str] = Field(default=None, description="Business location (city, state)")
    annual_revenue: Optional[float] = Field(default=None, description="Annual revenue in dollars")
    employee_count: Optional[int] = Field(default=None, description="Number of employees")
    years_in_business: Optional[int] = Field(default=None, description="Years company has been operating")
    coverage_requested: List[CoverageRequest] = Field(default_factory=list, description="List of coverages requested")
    vehicle_count: Optional[int] = Field(default=None, description="Number of vehicles if applicable")
    property_value: Optional[float] = Field(default=None, description="Property/equipment value if mentioned")
    loss_history: Optional[str] = Field(default=None, description="Description of claims/loss history")
    effective_date: Optional[str] = Field(default=None, description="Requested coverage start date")
    urgency: Optional[str] = Field(default=None, description="Urgency level mentioned")
    broker: Optional[BrokerContact] = Field(default=None, description="Broker contact information")
    raw_email: str = Field(description="Original email content")


class IndustryClassification(BaseModel):
    """Step 2: Industry classification result."""
    bic_code: str = Field(description="Business Industry Classification code")
    industry_name: str = Field(description="Industry name")
    risk_category: str = Field(description="Risk category (LOW, MEDIUM, HIGH)")
    confidence_score: float = Field(description="Classification confidence (0-1)")
    matching_keywords: List[str] = Field(default_factory=list, description="Keywords that matched")
    subcategory: Optional[str] = Field(default=None, description="Industry subcategory if applicable")


class RateInfo(BaseModel):
    """Step 3: Rate information from rating manuals."""
    bic_code: str
    coverage_type: str
    base_rate: float
    rate_basis: str = Field(description="What the rate applies to (revenue, payroll, vehicle_count, etc.)")
    minimum_premium: float
    source_document: Optional[str] = Field(default=None, description="Source document name")


class RevenueEstimate(BaseModel):
    """Step 4: Revenue estimation (if not provided)."""
    estimated_revenue: float
    estimation_method: str = Field(description="How revenue was estimated")
    confidence_level: str = Field(description="LOW, MEDIUM, HIGH")
    requires_verification: bool = Field(default=True, description="Flag for underwriter review")
    notes: Optional[str] = Field(default=None)


class PremiumLineItem(BaseModel):
    """Individual premium line item."""
    coverage_type: str
    base_premium: float
    rate_used: float
    rate_basis: str
    exposure_value: float = Field(description="Revenue, payroll, vehicle count, etc.")
    calculation_notes: Optional[str] = Field(default=None)


class PremiumCalculation(BaseModel):
    """Step 5: Base premium calculation."""
    line_items: List[PremiumLineItem] = Field(default_factory=list)
    total_base_premium: float
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModifierDetail(BaseModel):
    """Individual modifier applied."""
    modifier_name: str
    modifier_type: str
    modifier_value: float = Field(description="Percentage as decimal, e.g., -0.10 for 10% credit")
    reason: str
    premium_impact: float = Field(description="Dollar impact on premium")


class ModifierResult(BaseModel):
    """Step 6: Premium modifiers applied."""
    modifiers_applied: List[ModifierDetail] = Field(default_factory=list)
    total_modifier_impact: float = Field(description="Total dollar adjustment")
    total_modifier_percentage: float = Field(description="Combined modifier as percentage")
    adjusted_premium: float = Field(description="Premium after modifiers")


class AuthorityCheck(BaseModel):
    """Step 7: Authority level determination."""
    authority_level: str = Field(description="standard, senior, management, reinsurance")
    requires_approval: bool
    approval_reason: Optional[str] = Field(default=None)
    approver_role: Optional[str] = Field(default=None)
    auto_bind_eligible: bool = Field(default=False)
    referral_reasons: List[str] = Field(default_factory=list)


class EndorsementRecommendation(BaseModel):
    """Recommended coverage endorsement."""
    endorsement_name: str
    endorsement_type: str
    reason: str
    estimated_cost: Optional[float] = Field(default=None)
    required: bool = Field(default=False)


class CoverageLimitation(BaseModel):
    """Coverage limitation or exclusion."""
    limitation_type: str
    description: str
    reason: str


class CoverageAnalysis(BaseModel):
    """Step 8: Coverage analysis and recommendations."""
    recommended_endorsements: List[EndorsementRecommendation] = Field(default_factory=list)
    coverage_limitations: List[CoverageLimitation] = Field(default_factory=list)
    coverage_gaps: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(default=None)


class RiskFactor(BaseModel):
    """Individual risk factor identified."""
    factor_name: str
    factor_category: str
    severity: str = Field(description="LOW, MEDIUM, HIGH")
    description: str
    mitigation: Optional[str] = Field(default=None)


class RiskAssessment(BaseModel):
    """Step 9: Comprehensive risk assessment."""
    overall_risk_level: str = Field(description="LOW, MEDIUM, HIGH, VERY_HIGH")
    risk_score: float = Field(description="Numerical risk score 0-100")
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    positive_factors: List[str] = Field(default_factory=list)
    underwriting_notes: List[str] = Field(default_factory=list)
    recommendation: str = Field(description="ACCEPT, ACCEPT_WITH_CONDITIONS, REFER, DECLINE")


class QuotePremiumSummary(BaseModel):
    """Premium breakdown in the quote."""
    coverage_type: str
    premium: float
    limits: Optional[str] = Field(default=None)
    deductible: Optional[float] = Field(default=None)


class GeneratedQuote(BaseModel):
    """Step 10: Final generated quote."""
    quote_id: str
    client_name: str
    effective_date: Optional[str] = Field(default=None)
    expiration_date: Optional[str] = Field(default=None)
    quote_valid_until: str

    # Premium details
    premium_summary: List[QuotePremiumSummary] = Field(default_factory=list)
    total_annual_premium: float

    # Coverage details
    coverage_summary: str
    terms_and_conditions: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)

    # Underwriting notes
    underwriting_notes: List[str] = Field(default_factory=list)

    # Generated response
    quote_letter: str = Field(description="Full quote letter/email text")

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = Field(default="1.0")


class PipelineMetrics(BaseModel):
    """Metrics about the pipeline execution."""
    total_duration_seconds: float
    step_durations: Dict[str, float] = Field(default_factory=dict)
    llm_calls: int = Field(default=0)
    vector_searches: int = Field(default=0)
    documents_retrieved: int = Field(default=0)


class PipelineResult(BaseModel):
    """Complete result from the underwriting pipeline."""
    success: bool
    quote_id: Optional[str] = Field(default=None)

    # All step results
    extracted_email: Optional[ExtractedEmail] = Field(default=None)
    industry_classification: Optional[IndustryClassification] = Field(default=None)
    rate_info: List[RateInfo] = Field(default_factory=list)
    revenue_estimate: Optional[RevenueEstimate] = Field(default=None)
    premium_calculation: Optional[PremiumCalculation] = Field(default=None)
    modifier_result: Optional[ModifierResult] = Field(default=None)
    authority_check: Optional[AuthorityCheck] = Field(default=None)
    coverage_analysis: Optional[CoverageAnalysis] = Field(default=None)
    risk_assessment: Optional[RiskAssessment] = Field(default=None)
    generated_quote: Optional[GeneratedQuote] = Field(default=None)

    # Execution metadata
    metrics: Optional[PipelineMetrics] = Field(default=None)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
