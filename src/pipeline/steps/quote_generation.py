"""
Step 10: Quote Generation
Generates the final quote document and response email.
"""

import uuid
from datetime import datetime, timedelta
from typing import List
from src.core.fireworks_client import FireworksClient
from src.prompts import QUOTE_GENERATOR_SYSTEM, QUOTE_GENERATOR_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    PremiumCalculation,
    ModifierResult,
    CoverageAnalysis,
    RiskAssessment,
    AuthorityCheck,
    GeneratedQuote,
    QuotePremiumSummary,
)


class QuoteGenerationStep:
    """
    Generates the final quote document including the quote letter,
    premium summary, and all relevant details.
    """

    def __init__(self, llm_client: FireworksClient):
        """Initialize with LLM client."""
        self.llm_client = llm_client

    def execute(
        self,
        extracted_email: ExtractedEmail,
        industry: IndustryClassification,
        premium_calc: PremiumCalculation,
        modifier_result: ModifierResult,
        coverage_analysis: CoverageAnalysis,
        risk_assessment: RiskAssessment,
        authority_check: AuthorityCheck
    ) -> GeneratedQuote:
        """
        Generate the final quote.

        Args:
            All previous step results

        Returns:
            GeneratedQuote with complete quote details
        """
        # Generate quote ID
        quote_id = f"Q-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        # Build premium breakdown
        premium_summary = []
        for item in premium_calc.line_items:
            # Find coverage limits from original request
            limits = None
            for cov in extracted_email.coverage_requested:
                if cov.coverage_type.lower() in item.coverage_type.lower():
                    limits = cov.limits
                    break

            premium_summary.append(QuotePremiumSummary(
                coverage_type=item.coverage_type,
                premium=item.base_premium,
                limits=limits,
            ))

        # Build coverage summary
        coverage_lines = [
            f"- {item.coverage_type.replace('_', ' ').title()}: ${item.premium:,.2f}"
            + (f" ({item.limits})" if item.limits else "")
            for item in premium_summary
        ]
        coverage_summary_text = "\n".join(coverage_lines)

        # Build premium breakdown text
        premium_breakdown = self._build_premium_breakdown(
            premium_calc, modifier_result
        )

        # Build underwriting notes
        underwriting_notes = risk_assessment.underwriting_notes.copy()
        if authority_check.requires_approval:
            underwriting_notes.append(
                f"Requires {authority_check.approver_role} approval: {authority_check.approval_reason}"
            )

        # Build terms and conditions
        terms = self._build_terms_conditions(coverage_analysis, authority_check)

        # Format broker info
        broker_info = "Unknown Broker"
        if extracted_email.broker:
            broker_parts = []
            if extracted_email.broker.name:
                broker_parts.append(extracted_email.broker.name)
            if extracted_email.broker.brokerage:
                broker_parts.append(extracted_email.broker.brokerage)
            broker_info = ", ".join(broker_parts) if broker_parts else "Unknown Broker"

        # Generate quote letter using LLM
        prompt = QUOTE_GENERATOR_PROMPT.format(
            client_name=extracted_email.client_name,
            broker_info=broker_info,
            coverage_summary=coverage_summary_text,
            premium_breakdown=premium_breakdown,
            underwriting_notes="\n".join(underwriting_notes) or "Standard underwriting applies.",
            terms_conditions="\n".join(terms),
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=QUOTE_GENERATOR_SYSTEM,
            temperature=0.2,
        )

        # Determine dates
        effective_date = extracted_email.effective_date
        if not effective_date:
            # Default to first of next month
            today = datetime.now()
            if today.day > 15:
                effective_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
            else:
                effective_date = today.replace(day=1).strftime("%Y-%m-%d")

        expiration_date = None
        try:
            eff_dt = datetime.strptime(effective_date, "%Y-%m-%d")
            expiration_date = (eff_dt + timedelta(days=365)).strftime("%Y-%m-%d")
        except:
            pass

        quote_valid_until = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        return GeneratedQuote(
            quote_id=quote_id,
            client_name=extracted_email.client_name,
            effective_date=effective_date,
            expiration_date=expiration_date,
            quote_valid_until=quote_valid_until,
            premium_summary=premium_summary,
            total_annual_premium=modifier_result.adjusted_premium,
            coverage_summary=result.get("coverage_summary", coverage_summary_text),
            terms_and_conditions=result.get("terms_and_conditions", terms),
            exclusions=result.get("exclusions", []),
            underwriting_notes=underwriting_notes,
            quote_letter=result.get("quote_letter", self._fallback_quote_letter(
                extracted_email, broker_info, premium_breakdown, modifier_result, quote_id
            )),
            generated_at=datetime.utcnow(),
        )

    def _build_premium_breakdown(
        self,
        premium_calc: PremiumCalculation,
        modifier_result: ModifierResult
    ) -> str:
        """Build premium breakdown text."""
        lines = ["Premium Breakdown:"]

        for item in premium_calc.line_items:
            lines.append(
                f"  {item.coverage_type.replace('_', ' ').title()}: ${item.base_premium:,.2f}"
            )

        lines.append(f"\nBase Premium Total: ${premium_calc.total_base_premium:,.2f}")

        if modifier_result.modifiers_applied:
            lines.append("\nModifiers Applied:")
            for mod in modifier_result.modifiers_applied:
                sign = "+" if mod.premium_impact > 0 else ""
                lines.append(
                    f"  {mod.modifier_name}: {sign}${mod.premium_impact:,.2f} "
                    f"({mod.modifier_value:+.0%})"
                )

        lines.append(f"\nTotal Annual Premium: ${modifier_result.adjusted_premium:,.2f}")

        return "\n".join(lines)

    def _build_terms_conditions(
        self,
        coverage_analysis: CoverageAnalysis,
        authority_check: AuthorityCheck
    ) -> List[str]:
        """Build terms and conditions list."""
        terms = [
            "Quote valid for 30 days from issue date",
            "Subject to receipt of signed application",
            "Subject to verification of information provided",
            "Premium subject to audit",
        ]

        if authority_check.requires_approval:
            terms.append(f"Subject to {authority_check.approver_role} approval")

        for endorsement in coverage_analysis.recommended_endorsements:
            if endorsement.required:
                terms.append(f"Required endorsement: {endorsement.endorsement_name}")

        return terms

    def _fallback_quote_letter(
        self,
        extracted_email: ExtractedEmail,
        broker_info: str,
        premium_breakdown: str,
        modifier_result: ModifierResult,
        quote_id: str
    ) -> str:
        """Generate fallback quote letter if LLM fails."""
        return f"""Subject: Quote {quote_id} for {extracted_email.client_name}

Dear {broker_info},

Thank you for your quote request for {extracted_email.client_name}. We are pleased to provide the following quotation:

{premium_breakdown}

This quote is valid for 30 days from the date of issue.

To proceed with binding coverage, please:
1. Review the quote details and confirm accuracy
2. Complete and sign the application
3. Provide any additional documentation requested
4. Remit the required down payment

Please don't hesitate to contact us if you have any questions.

Best regards,
Underwriting Department
"""
