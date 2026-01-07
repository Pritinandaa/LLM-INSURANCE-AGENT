"""
Step 4: Revenue Estimation
Estimates annual revenue when not provided in the email.
"""

from typing import Optional
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import REVENUE_ESTIMATOR_SYSTEM, REVENUE_ESTIMATOR_PROMPT
from src.pipeline.models import ExtractedEmail, IndustryClassification, RevenueEstimate


class RevenueEstimationStep:
    """
    Estimates annual revenue based on industry benchmarks and available data.
    Only used when revenue is not provided in the original email.
    """

    # Industry average revenue per employee (rough estimates)
    INDUSTRY_REVENUE_PER_EMPLOYEE = {
        "11": 150000,   # Agriculture
        "21": 500000,   # Mining
        "22": 400000,   # Utilities
        "23": 180000,   # Construction - General
        "44": 200000,   # Construction - Commercial
        "31": 250000,   # Manufacturing - Food
        "42": 350000,   # Wholesale Trade
        "44-45": 150000,  # Retail Trade
        "48": 120000,   # Trucking
        "51": 300000,   # IT
        "54": 200000,   # Professional Services
        "62": 180000,   # Healthcare
        "72-2": 70000,  # Restaurants
        "81-2": 60000,  # Personal Care
    }

    def __init__(
        self,
        llm_client: FireworksClient,
        vector_search: VectorSearchService
    ):
        """Initialize with required services."""
        self.llm_client = llm_client
        self.vector_search = vector_search

    def execute(
        self,
        extracted_email: ExtractedEmail,
        industry: IndustryClassification
    ) -> Optional[RevenueEstimate]:
        """
        Estimate revenue if not provided.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification

        Returns:
            RevenueEstimate if revenue was estimated, None if already provided
        """
        # If revenue was provided, no estimation needed
        if extracted_email.annual_revenue:
            return None

        # If no employee count, use a simple estimate
        if not extracted_email.employee_count:
            return self._simple_estimate(industry)

        # Use employee-based estimation
        return self._employee_based_estimate(extracted_email, industry)

    def _employee_based_estimate(
        self,
        extracted_email: ExtractedEmail,
        industry: IndustryClassification
    ) -> RevenueEstimate:
        """Estimate revenue based on employee count and industry."""
        # Get industry revenue per employee
        rev_per_employee = self.INDUSTRY_REVENUE_PER_EMPLOYEE.get(
            industry.bic_code,
            150000  # Default
        )

        estimated_revenue = extracted_email.employee_count * rev_per_employee

        # Adjust for years in business (established companies tend to have higher revenue)
        if extracted_email.years_in_business:
            if extracted_email.years_in_business > 10:
                estimated_revenue *= 1.2
            elif extracted_email.years_in_business > 5:
                estimated_revenue *= 1.1

        return RevenueEstimate(
            estimated_revenue=estimated_revenue,
            estimation_method="employee_count_multiplier",
            confidence_level="MEDIUM",
            requires_verification=True,
            notes=f"Estimated based on {extracted_email.employee_count} employees "
                  f"in {industry.industry_name} industry. "
                  f"Using ${rev_per_employee:,.0f} revenue per employee benchmark.",
        )

    def _simple_estimate(self, industry: IndustryClassification) -> RevenueEstimate:
        """Simple estimate when minimal information is available."""
        # Use industry median estimates
        industry_medians = {
            "11": 500000,    # Agriculture
            "21": 2000000,   # Mining
            "22": 1500000,   # Utilities
            "23": 750000,    # Construction - General
            "44": 1500000,   # Construction - Commercial
            "31": 2000000,   # Manufacturing
            "42": 3000000,   # Wholesale
            "44-45": 1000000,  # Retail
            "48": 1000000,   # Trucking
            "51": 2000000,   # IT
            "54": 750000,    # Professional Services
            "62": 1500000,   # Healthcare
            "72-2": 500000,  # Restaurants
            "81-2": 250000,  # Personal Care
        }

        estimated_revenue = industry_medians.get(industry.bic_code, 1000000)

        return RevenueEstimate(
            estimated_revenue=estimated_revenue,
            estimation_method="industry_median",
            confidence_level="LOW",
            requires_verification=True,
            notes=f"Estimated using industry median for {industry.industry_name}. "
                  f"Limited information available - underwriter verification required.",
        )
