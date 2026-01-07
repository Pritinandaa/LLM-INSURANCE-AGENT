"""
Step 8: Coverage Analysis
Analyzes coverage needs and recommends endorsements.
"""

from typing import List
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import COVERAGE_ANALYSIS_SYSTEM, COVERAGE_ANALYSIS_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    CoverageAnalysis,
    EndorsementRecommendation,
    CoverageLimitation,
)


class CoverageAnalysisStep:
    """
    Analyzes coverage needs and recommends endorsements,
    identifies gaps, and notes limitations.
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
    ) -> CoverageAnalysis:
        """
        Analyze coverage needs and make recommendations.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification

        Returns:
            CoverageAnalysis with recommendations
        """
        # Search for coverage guidelines
        search_query = (
            f"Coverage endorsements {industry.industry_name} "
            f"recommendations requirements"
        )

        rag_result = self.vector_search.rag_query(
            collection_name=Collections.UNDERWRITING_GUIDELINES,
            query=search_query,
            limit=5,
        )

        # Format requested coverages
        requested = ", ".join([
            f"{c.coverage_type} ({c.limits or 'TBD'})"
            for c in extracted_email.coverage_requested
        ]) if extracted_email.coverage_requested else "General Liability, Property"

        # Build operations details
        operations = []
        if extracted_email.employee_count:
            operations.append(f"{extracted_email.employee_count} employees")
        if extracted_email.vehicle_count:
            operations.append(f"{extracted_email.vehicle_count} vehicles")
        if extracted_email.location:
            operations.append(f"Location: {extracted_email.location}")
        operations_str = ", ".join(operations) if operations else "Standard operations"

        prompt = COVERAGE_ANALYSIS_PROMPT.format(
            business_description=extracted_email.industry_description,
            industry=industry.industry_name,
            operations_details=operations_str,
            requested_coverages=requested,
            guidelines_context=rag_result["context"],
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=COVERAGE_ANALYSIS_SYSTEM,
            temperature=0.1,
        )

        # Parse endorsements
        endorsements = []
        for end_data in result.get("recommended_endorsements", []):
            if isinstance(end_data, dict):
                # Safely parse estimated_cost - may be string, number, or None
                raw_cost = end_data.get("estimated_cost")
                estimated_cost = None
                if raw_cost is not None:
                    if isinstance(raw_cost, (int, float)):
                        estimated_cost = float(raw_cost)
                    elif isinstance(raw_cost, str):
                        # Try to extract numeric value from string
                        import re
                        numbers = re.findall(r'[\d.]+', raw_cost)
                        if numbers:
                            try:
                                estimated_cost = float(numbers[0])
                            except ValueError:
                                estimated_cost = None

                endorsements.append(EndorsementRecommendation(
                    endorsement_name=end_data.get("endorsement_name", ""),
                    endorsement_type=end_data.get("endorsement_type", "optional"),
                    reason=end_data.get("reason", ""),
                    estimated_cost=estimated_cost,
                    required=end_data.get("required", False),
                ))

        # Parse limitations
        limitations = []
        for lim_data in result.get("coverage_limitations", []):
            if isinstance(lim_data, dict):
                limitations.append(CoverageLimitation(
                    limitation_type=lim_data.get("limitation_type", "exclusion"),
                    description=lim_data.get("description", ""),
                    reason=lim_data.get("reason", ""),
                ))

        # Handle notes - may be string or list
        raw_notes = result.get("notes")
        if isinstance(raw_notes, list):
            notes = " ".join(str(n) for n in raw_notes)
        elif raw_notes:
            notes = str(raw_notes)
        else:
            notes = None

        # Handle coverage_gaps - may be list of strings or list of dicts
        raw_gaps = result.get("coverage_gaps", [])
        coverage_gaps = []
        for item in raw_gaps:
            if isinstance(item, str):
                coverage_gaps.append(item)
            elif isinstance(item, dict):
                # Extract gap name or description from dict
                gap = item.get("gap") or item.get("name") or item.get("description", "")
                if gap:
                    coverage_gaps.append(str(gap))

        return CoverageAnalysis(
            recommended_endorsements=endorsements,
            coverage_limitations=limitations,
            coverage_gaps=coverage_gaps,
            notes=notes,
        )
