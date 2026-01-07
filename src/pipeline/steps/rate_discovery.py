"""
Step 3: Rate Discovery
Retrieves applicable base rates from rating manuals.
"""

from typing import List
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import RATE_DISCOVERY_SYSTEM, RATE_DISCOVERY_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    RateInfo,
    CoverageRequest,
)


class RateDiscoveryStep:
    """
    Discovers applicable base rates from rating manuals.
    Uses vector search to find relevant rate tables.
    """

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
    ) -> List[RateInfo]:
        """
        Find applicable rates for the requested coverages.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification

        Returns:
            List of RateInfo objects for each coverage
        """
        # Format coverage list
        coverages = ", ".join([
            cov.coverage_type for cov in extracted_email.coverage_requested
        ]) if extracted_email.coverage_requested else "general_liability, property"

        # Search for rating information
        search_query = f"Base rates {industry.industry_name} BIC {industry.bic_code} {coverages}"

        rag_result = self.vector_search.rag_query(
            collection_name=Collections.RATING_MANUALS,
            query=search_query,
            limit=5,
        )

        # Format prompt
        prompt = RATE_DISCOVERY_PROMPT.format(
            industry_name=industry.industry_name,
            bic_code=industry.bic_code,
            coverages=coverages,
            rating_context=rag_result["context"],
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=RATE_DISCOVERY_SYSTEM,
            temperature=0.1,
        )

        # Parse rate info
        rates = []
        rate_items = result.get("rate_info", result.get("rates", []))
        if isinstance(rate_items, list):
            for rate_data in rate_items:
                if isinstance(rate_data, dict):
                    # Safely parse numeric values with defaults
                    base_rate = rate_data.get("base_rate") or 0
                    min_premium = rate_data.get("minimum_premium") or 1000
                    rates.append(RateInfo(
                        bic_code=rate_data.get("bic_code") or industry.bic_code,
                        coverage_type=rate_data.get("coverage_type") or "unknown",
                        base_rate=float(base_rate),
                        rate_basis=rate_data.get("rate_basis") or "per_1000_revenue",
                        minimum_premium=float(min_premium),
                        source_document=rate_data.get("source_document"),
                    ))

        # If no rates found, provide defaults based on coverage requested
        if not rates:
            rates = self._get_default_rates(extracted_email, industry)

        return rates

    def _get_default_rates(
        self,
        extracted_email: ExtractedEmail,
        industry: IndustryClassification
    ) -> List[RateInfo]:
        """Provide default rates if none found in search."""
        default_rates = []

        # Default GL rate
        default_rates.append(RateInfo(
            bic_code=industry.bic_code,
            coverage_type="general_liability",
            base_rate=5.0,
            rate_basis="per_1000_revenue",
            minimum_premium=1500,
        ))

        # Add property if mentioned
        if extracted_email.property_value:
            default_rates.append(RateInfo(
                bic_code=industry.bic_code,
                coverage_type="property",
                base_rate=0.35,
                rate_basis="percent_of_tiv",
                minimum_premium=1000,
            ))

        # Add auto if vehicles mentioned
        if extracted_email.vehicle_count:
            default_rates.append(RateInfo(
                bic_code=industry.bic_code,
                coverage_type="auto_liability",
                base_rate=750,
                rate_basis="per_vehicle",
                minimum_premium=1500,
            ))

        return default_rates
