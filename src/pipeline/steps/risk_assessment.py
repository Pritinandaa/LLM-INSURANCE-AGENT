"""
Step 9: Risk Assessment
Conducts comprehensive risk assessment for the account.
"""

from typing import List
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import RISK_ASSESSMENT_SYSTEM, RISK_ASSESSMENT_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    ModifierResult,
    RiskAssessment,
    RiskFactor,
)


class RiskAssessmentStep:
    """
    Conducts comprehensive risk assessment using all available data.
    Produces risk score, factors, and underwriting recommendation.
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
        industry: IndustryClassification,
        modifier_result: ModifierResult
    ) -> RiskAssessment:
        """
        Conduct comprehensive risk assessment.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification
            modifier_result: Premium modification details

        Returns:
            RiskAssessment with overall evaluation
        """
        # Search for risk assessment guidelines
        search_query = (
            f"Risk assessment underwriting {industry.industry_name} "
            f"appetite guidelines factors"
        )

        rag_result = self.vector_search.rag_query(
            collection_name=Collections.UNDERWRITING_GUIDELINES,
            query=search_query,
            limit=5,
        )

        # Build client profile
        client_profile = self._build_client_profile(extracted_email, modifier_result)

        # Industry analysis
        industry_analysis = (
            f"Industry: {industry.industry_name}\n"
            f"BIC Code: {industry.bic_code}\n"
            f"Risk Category: {industry.risk_category}\n"
            f"Classification Confidence: {industry.confidence_score:.0%}"
        )

        prompt = RISK_ASSESSMENT_PROMPT.format(
            client_profile=client_profile,
            industry_analysis=industry_analysis,
            loss_history=extracted_email.loss_history or "No loss history provided",
            guidelines_context=rag_result["context"],
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=RISK_ASSESSMENT_SYSTEM,
            temperature=0.1,
        )

        # Parse risk factors
        risk_factors = []
        for factor_data in result.get("risk_factors", []):
            if isinstance(factor_data, dict):
                risk_factors.append(RiskFactor(
                    factor_name=factor_data.get("factor_name", ""),
                    factor_category=factor_data.get("factor_category", "general"),
                    severity=factor_data.get("severity", "MEDIUM"),
                    description=factor_data.get("description", ""),
                    mitigation=factor_data.get("mitigation"),
                ))

        # Parse positive_factors - may be list of strings or list of dicts
        raw_positive = result.get("positive_factors", [])
        positive_factors = []
        for item in raw_positive:
            if isinstance(item, str):
                positive_factors.append(item)
            elif isinstance(item, dict):
                # Extract factor name or description from dict
                name = item.get("factor_name") or item.get("name") or item.get("description", "")
                if name:
                    positive_factors.append(str(name))

        # Parse underwriting_notes - may be list of strings or list of dicts
        raw_notes = result.get("underwriting_notes", [])
        underwriting_notes = []
        for item in raw_notes:
            if isinstance(item, str):
                underwriting_notes.append(item)
            elif isinstance(item, dict):
                note = item.get("note") or item.get("description") or item.get("text", "")
                if note:
                    underwriting_notes.append(str(note))

        # Safely get risk_score
        raw_score = result.get("risk_score", 50)
        try:
            risk_score = float(raw_score) if raw_score is not None else 50.0
        except (ValueError, TypeError):
            risk_score = 50.0

        return RiskAssessment(
            overall_risk_level=result.get("overall_risk_level", "MEDIUM"),
            risk_score=risk_score,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
            underwriting_notes=underwriting_notes,
            recommendation=result.get("recommendation", "ACCEPT"),
        )

    def _build_client_profile(
        self,
        extracted_email: ExtractedEmail,
        modifier_result: ModifierResult
    ) -> str:
        """Build formatted client profile string."""
        lines = [
            f"Client: {extracted_email.client_name}",
            f"Industry: {extracted_email.industry_description}",
        ]

        if extracted_email.location:
            lines.append(f"Location: {extracted_email.location}")

        if extracted_email.annual_revenue:
            lines.append(f"Annual Revenue: ${extracted_email.annual_revenue:,.0f}")

        if extracted_email.employee_count:
            lines.append(f"Employees: {extracted_email.employee_count}")

        if extracted_email.years_in_business:
            lines.append(f"Years in Business: {extracted_email.years_in_business}")

        if extracted_email.vehicle_count:
            lines.append(f"Vehicles: {extracted_email.vehicle_count}")

        lines.append(f"Adjusted Premium: ${modifier_result.adjusted_premium:,.2f}")

        if modifier_result.modifiers_applied:
            mod_summary = ", ".join([
                f"{m.modifier_name}: {m.modifier_value:+.0%}"
                for m in modifier_result.modifiers_applied
            ])
            lines.append(f"Modifiers Applied: {mod_summary}")

        return "\n".join(lines)
