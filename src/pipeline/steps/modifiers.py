"""
Step 6: Premium Modifiers
Applies credits and debits based on risk characteristics.
"""

from typing import List
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import MODIFIER_ANALYSIS_SYSTEM, MODIFIER_ANALYSIS_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    PremiumCalculation,
    ModifierResult,
    ModifierDetail,
)


class ModifiersStep:
    """
    Analyzes risk factors and applies premium modifiers.
    Uses vector search to find applicable modifier rules.
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
        premium_calc: PremiumCalculation
    ) -> ModifierResult:
        """
        Determine and apply premium modifiers.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification
            premium_calc: Base premium calculation

        Returns:
            ModifierResult with all applied modifiers
        """
        # Search for applicable modifiers
        search_query = (
            f"Premium modifiers credits debits "
            f"{industry.industry_name} loss history years in business"
        )

        rag_result = self.vector_search.rag_query(
            collection_name=Collections.MODIFIERS,
            query=search_query,
            limit=5,
        )

        # Format prompt
        prompt = MODIFIER_ANALYSIS_PROMPT.format(
            industry=industry.industry_name,
            years_in_business=extracted_email.years_in_business or "Unknown",
            loss_history=extracted_email.loss_history or "No loss history provided",
            employee_count=extracted_email.employee_count or "Unknown",
            vehicle_count=extracted_email.vehicle_count or 0,
            location=extracted_email.location or "Unknown",
            base_premium=premium_calc.total_base_premium,
            modifiers_context=rag_result["context"],
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=MODIFIER_ANALYSIS_SYSTEM,
            temperature=0.1,
        )

        # Parse modifiers
        modifiers = []
        for mod_data in result.get("modifiers_applied", []):
            if isinstance(mod_data, dict):
                modifiers.append(ModifierDetail(
                    modifier_name=mod_data.get("modifier_name", "Unknown"),
                    modifier_type=mod_data.get("modifier_type", "experience"),
                    modifier_value=float(mod_data.get("modifier_value", 0)),
                    reason=mod_data.get("reason", ""),
                    premium_impact=float(mod_data.get("premium_impact", 0)),
                ))

        # Calculate totals
        total_impact = sum(m.premium_impact for m in modifiers)
        total_percentage = sum(m.modifier_value for m in modifiers)
        adjusted_premium = premium_calc.total_base_premium + total_impact

        return ModifierResult(
            modifiers_applied=modifiers,
            total_modifier_impact=round(total_impact, 2),
            total_modifier_percentage=round(total_percentage, 4),
            adjusted_premium=round(adjusted_premium, 2),
        )
