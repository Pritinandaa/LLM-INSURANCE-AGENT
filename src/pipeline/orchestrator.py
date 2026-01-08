"""
Pipeline Orchestrator
Coordinates all 10 steps of the underwriting pipeline.
"""

import time
import logging
from typing import Optional, Callable
from datetime import datetime

from src.core.fireworks_client import get_fireworks_client, FireworksClient
from src.core.vector_search import get_vector_search_service, VectorSearchService
from src.core.mongodb_client import get_collection, Collections

from src.pipeline.models import (
    PipelineResult,
    PipelineMetrics,
    ExtractedEmail,
)
from src.pipeline.steps import (
    EmailParserStep,
    IndustryClassifierStep,
    RateDiscoveryStep,
    RevenueEstimationStep,
    PremiumCalculationStep,
    ModifiersStep,
    AuthorityCheckStep,
    CoverageAnalysisStep,
    RiskAssessmentStep,
    QuoteGenerationStep,
)


logger = logging.getLogger(__name__)


class UnderwritingPipeline:
    """
    Orchestrates the 10-step underwriting pipeline.
    Handles execution, error recovery, and progress tracking.
    """

    def __init__(
        self,
        llm_client: Optional[FireworksClient] = None,
        vector_search: Optional[VectorSearchService] = None,
        progress_callback: Optional[Callable[[int, str, str], None]] = None
    ):
        """
        Initialize the pipeline with required services.

        Args:
            llm_client: Fireworks client (uses default if not provided)
            vector_search: Vector search service (uses default if not provided)
            progress_callback: Optional callback for progress updates
                             (step_number, step_name, status)
        """
        self.llm_client = llm_client or get_fireworks_client()
        self.vector_search = vector_search or get_vector_search_service()
        self.progress_callback = progress_callback

        # Initialize all steps
        self.steps = {
            "email_parser": EmailParserStep(self.llm_client),
            "industry_classifier": IndustryClassifierStep(
                self.llm_client, self.vector_search
            ),
            "rate_discovery": RateDiscoveryStep(
                self.llm_client, self.vector_search
            ),
            "revenue_estimation": RevenueEstimationStep(
                self.llm_client, self.vector_search
            ),
            "premium_calculation": PremiumCalculationStep(),
            "modifiers": ModifiersStep(
                self.llm_client, self.vector_search
            ),
            "authority_check": AuthorityCheckStep(
                self.llm_client, self.vector_search
            ),
            "coverage_analysis": CoverageAnalysisStep(
                self.llm_client, self.vector_search
            ),
            "risk_assessment": RiskAssessmentStep(
                self.llm_client, self.vector_search
            ),
            "quote_generation": QuoteGenerationStep(self.llm_client),
        }

    def _report_progress(self, step: int, name: str, status: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(step, name, status)

    def process(self, email_content: str) -> PipelineResult:
        """
        Process an email through the complete underwriting pipeline.

        Args:
            email_content: Raw email text

        Returns:
            PipelineResult with all step results and final quote
        """
        start_time = time.time()
        step_durations = {}
        errors = []
        warnings = []
        llm_calls = 0
        vector_searches = 0
        documents_retrieved = 0

        result = PipelineResult(success=False)

        try:
            # Step 1: Parse Email
            self._report_progress(1, "Email Parser", "running")
            step_start = time.time()
            extracted_email = self.steps["email_parser"].execute(email_content)
            step_durations["email_parser"] = time.time() - step_start
            result.extracted_email = extracted_email
            llm_calls += 1
            self._report_progress(1, "Email Parser", "complete")
            logger.info(f"Step 1 complete: Extracted info for {extracted_email.client_name}")

            # Step 2: Classify Industry
            self._report_progress(2, "Industry Classifier", "running")
            step_start = time.time()
            industry = self.steps["industry_classifier"].execute(extracted_email)
            step_durations["industry_classifier"] = time.time() - step_start
            result.industry_classification = industry
            llm_calls += 1
            vector_searches += 1
            self._report_progress(2, "Industry Classifier", "complete")
            logger.info(f"Step 2 complete: BIC Code {industry.bic_code} - {industry.industry_name}")

            # Step 3: Rate Discovery
            self._report_progress(3, "Rate Discovery", "running")
            step_start = time.time()
            rates = self.steps["rate_discovery"].execute(extracted_email, industry)
            step_durations["rate_discovery"] = time.time() - step_start
            result.rate_info = rates
            llm_calls += 1
            vector_searches += 1
            self._report_progress(3, "Rate Discovery", "complete")
            logger.info(f"Step 3 complete: Found {len(rates)} applicable rates")

            # Step 4: Revenue Estimation (if needed)
            self._report_progress(4, "Revenue Estimation", "running")
            step_start = time.time()
            revenue_estimate = self.steps["revenue_estimation"].execute(
                extracted_email, industry
            )
            step_durations["revenue_estimation"] = time.time() - step_start
            result.revenue_estimate = revenue_estimate
            if revenue_estimate:
                warnings.append(f"Revenue estimated at ${revenue_estimate.estimated_revenue:,.0f} - requires verification")
            self._report_progress(4, "Revenue Estimation", "complete")
            logger.info(f"Step 4 complete: Revenue {'estimated' if revenue_estimate else 'provided'}")

            # Step 5: Premium Calculation
            self._report_progress(5, "Premium Calculation", "running")
            step_start = time.time()
            premium_calc = self.steps["premium_calculation"].execute(
                extracted_email, rates, revenue_estimate
            )
            step_durations["premium_calculation"] = time.time() - step_start
            result.premium_calculation = premium_calc
            self._report_progress(5, "Premium Calculation", "complete")
            logger.info(f"Step 5 complete: Base premium ${premium_calc.total_base_premium:,.2f}")

            # Step 6: Apply Modifiers
            self._report_progress(6, "Modifiers", "running")
            step_start = time.time()
            modifier_result = self.steps["modifiers"].execute(
                extracted_email, industry, premium_calc
            )
            step_durations["modifiers"] = time.time() - step_start
            result.modifier_result = modifier_result
            llm_calls += 1
            vector_searches += 1
            self._report_progress(6, "Modifiers", "complete")
            logger.info(f"Step 6 complete: Adjusted premium ${modifier_result.adjusted_premium:,.2f}")

            # Step 7: Authority Check
            self._report_progress(7, "Authority Check", "running")
            step_start = time.time()
            authority = self.steps["authority_check"].execute(
                extracted_email, industry, modifier_result
            )
            step_durations["authority_check"] = time.time() - step_start
            result.authority_check = authority
            if authority.requires_approval:
                warnings.append(f"Requires {authority.approver_role} approval")
            self._report_progress(7, "Authority Check", "complete")
            logger.info(f"Step 7 complete: Authority level - {authority.authority_level}")

            # Step 8: Coverage Analysis
            self._report_progress(8, "Coverage Analysis", "running")
            step_start = time.time()
            coverage = self.steps["coverage_analysis"].execute(
                extracted_email, industry
            )
            step_durations["coverage_analysis"] = time.time() - step_start
            result.coverage_analysis = coverage
            llm_calls += 1
            vector_searches += 1
            self._report_progress(8, "Coverage Analysis", "complete")
            logger.info(f"Step 8 complete: {len(coverage.recommended_endorsements)} endorsements recommended")

            # Step 9: Risk Assessment
            self._report_progress(9, "Risk Assessment", "running")
            step_start = time.time()
            risk = self.steps["risk_assessment"].execute(
                extracted_email, industry, modifier_result
            )
            step_durations["risk_assessment"] = time.time() - step_start
            result.risk_assessment = risk
            llm_calls += 1
            vector_searches += 1
            self._report_progress(9, "Risk Assessment", "complete")
            logger.info(f"Step 9 complete: Risk level - {risk.overall_risk_level}, Score: {risk.risk_score}")

            # Step 10: Generate Quote
            self._report_progress(10, "Quote Generation", "running")
            step_start = time.time()
            quote = self.steps["quote_generation"].execute(
                extracted_email,
                industry,
                premium_calc,
                modifier_result,
                coverage,
                risk,
                authority
            )
            step_durations["quote_generation"] = time.time() - step_start
            result.generated_quote = quote
            result.quote_id = quote.quote_id
            llm_calls += 1
            self._report_progress(10, "Quote Generation", "complete")
            logger.info(f"Step 10 complete: Quote {quote.quote_id} generated")

            # Mark success
            result.success = True

            # Save to MongoDB
            self._save_result(result)

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            errors.append(str(e))
            result.success = False

        # Record metrics
        total_duration = time.time() - start_time
        result.metrics = PipelineMetrics(
            total_duration_seconds=round(total_duration, 2),
            step_durations={k: round(v, 2) for k, v in step_durations.items()},
            llm_calls=llm_calls,
            vector_searches=vector_searches,
            documents_retrieved=documents_retrieved,
        )
        result.errors = errors
        result.warnings = warnings

        return result

    def _save_result(self, result: PipelineResult):
        """Save the pipeline result to MongoDB."""
        try:
            quotes_collection = get_collection(Collections.QUOTES)

            # Convert to dict and save
            result_dict = result.model_dump()
            result_dict["_id"] = result.quote_id
            result_dict["saved_at"] = datetime.utcnow()

            quotes_collection.replace_one(
                {"_id": result.quote_id},
                result_dict,
                upsert=True
            )
            logger.info(f"Quote {result.quote_id} saved to MongoDB")
        except Exception as e:
            logger.error(f"Failed to save quote to MongoDB: {e}")
