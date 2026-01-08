"""
Step 7: Authority Check
Determines the underwriting authority level required for the quote.
"""

from typing import List
from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import AUTHORITY_CHECK_PROMPT
from src.pipeline.models import (
    ExtractedEmail,
    IndustryClassification,
    ModifierResult,
    AuthorityCheck,
)


class AuthorityCheckStep:
    """
    Determines the approval level needed based on premium size,
    industry risk, and other factors.
    """

    # Authority thresholds
    STANDARD_MAX_PREMIUM = 50000
    SENIOR_MAX_PREMIUM = 150000
    MANAGEMENT_MAX_PREMIUM = 500000

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
    ) -> AuthorityCheck:
        """
        Determine authority level required.

        Args:
            extracted_email: Parsed email data
            industry: Industry classification
            modifier_result: Premium after modifiers

        Returns:
            AuthorityCheck with authority level and approval requirements
        """
        premium = modifier_result.adjusted_premium
        referral_reasons = []

        # Determine base authority level from premium
        if premium <= self.STANDARD_MAX_PREMIUM:
            authority_level = "standard"
        elif premium <= self.SENIOR_MAX_PREMIUM:
            authority_level = "senior"
            referral_reasons.append(f"Premium ${premium:,.0f} exceeds standard authority")
        elif premium <= self.MANAGEMENT_MAX_PREMIUM:
            authority_level = "management"
            referral_reasons.append(f"Premium ${premium:,.0f} requires management approval")
        else:
            authority_level = "reinsurance"
            referral_reasons.append(f"Premium ${premium:,.0f} exceeds treaty limits")

        # Check for high-risk industry
        if industry.risk_category == "HIGH":
            if authority_level == "standard":
                authority_level = "senior"
            referral_reasons.append(f"High-risk industry: {industry.industry_name}")

        # Check loss history
        if extracted_email.loss_history:
            loss_lower = extracted_email.loss_history.lower()
            if any(word in loss_lower for word in ["multiple", "several", "frequent", "large", "major"]):
                if authority_level in ["standard", "senior"]:
                    authority_level = "management" if authority_level == "senior" else "senior"
                referral_reasons.append("Adverse loss history requires review")

        # Check for new business
        if extracted_email.years_in_business and extracted_email.years_in_business < 2:
            referral_reasons.append("New venture - less than 2 years in business")

        # Determine if approval is required
        requires_approval = authority_level != "standard"
        auto_bind_eligible = (
            authority_level == "standard" and
            industry.risk_category != "HIGH" and
            len(referral_reasons) == 0
        )

        # Set approver role
        approver_role = None
        if authority_level == "senior":
            approver_role = "Senior Underwriter"
        elif authority_level == "management":
            approver_role = "Underwriting Manager"
        elif authority_level == "reinsurance":
            approver_role = "Reinsurance Team"

        approval_reason = "; ".join(referral_reasons) if referral_reasons else None

        return AuthorityCheck(
            authority_level=authority_level,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
            approver_role=approver_role,
            auto_bind_eligible=auto_bind_eligible,
            referral_reasons=referral_reasons,
        )
