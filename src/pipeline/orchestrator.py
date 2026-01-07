"""
Pipeline Orchestrator
Coordinates the underwriting pipeline using exclusively Microsoft AutoGen.
"""

import time
import logging
from typing import Optional, Callable
from datetime import datetime

# Strict AutoGen imports
try:
    from src.core.autogen_client import get_autogen_client, process_underwriting_request_sync, AutoGenClient
except ImportError as e:
    raise ImportError(f"Failed to import AutoGen client: {e}. Check dependencies.")

from src.core.mongodb_client import get_collection, Collections
from src.pipeline.models import (
    PipelineResult,
    PipelineMetrics,
    ExtractedEmail,
)

logger = logging.getLogger(__name__)


class UnderwritingPipeline:
    """
    Orchestrates the underwriting pipeline using Microsoft AutoGen agents.
    Traditional sequential pipeline has been deprecated and removed.
    """

    def __init__(
        self,
        autogen_client: Optional[AutoGenClient] = None,
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        use_autogen: bool = True # Deprecated arg, kept for compatibility signatures but ignored
    ):
        """
        Initialize the pipeline.

        Args:
            autogen_client: AutoGen client (uses default if not provided)
            progress_callback: Optional callback for progress updates
            use_autogen: Ignored (Always True)
        """
        self.autogen_client = autogen_client or get_autogen_client()
        self.progress_callback = progress_callback
        
        logger.info("🚀 Initialized UnderwritingPipeline with Microsoft AutoGen (Vertex AI)")

    def _report_progress(self, step: int, name: str, status: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(step, name, status)

    def process(self, email_content: str) -> PipelineResult:
        """
        Process an email through the AutoGen underwriting pipeline.

        Args:
            email_content: Raw email text

        Returns:
            PipelineResult with quote details
        """
        logger.info("🔄 Using AutoGen agents with Vertex AI for processing")
        return self._process_with_autogen(email_content)

    def _process_with_autogen(self, email_content: str) -> PipelineResult:
        """
        Process email using Microsoft AutoGen agents.
        """
        start_time = time.time()
        result = PipelineResult(success=False)

        try:
            self._report_progress(1, "Microsoft AutoGen Pipeline", "running")
            logger.info("🤖 Starting Microsoft AutoGen multi-agent processing")

            # Use the synchronous wrapper for AutoGen processing
            autogen_result = process_underwriting_request_sync(email_content)

            if autogen_result.get("status") == "processed":
                result.success = True
                result.quote_id = f"autogen-{int(start_time)}"
                result.warnings = ["Microsoft AutoGen processing completed"]

                # Try to extract structured data from the AutoGen response
                with open("debug_dump.txt", "w", encoding="utf-8") as f:
                    f.write(str(autogen_result))
                
                quote_content = autogen_result.get("quote", "")
                
                # DEBUG: Log the raw content
                logger.error(f"DEBUG RAW AUTOGEN OUTPUT: {quote_content}")
                print(f"DEBUG RAW AUTOGEN OUTPUT: {quote_content}")

                from src.utils.json_utils import extract_json_from_text
                quote_data = extract_json_from_text(quote_content)

                if quote_data:
                    # Populate result fields from parsed data
                    # Keys are already normalized to snake_case by the utility
                    
                    if "client_name" in quote_data:
                        result.extracted_email = ExtractedEmail(
                            client_name=quote_data["client_name"],
                            industry_description="Extracted by AutoGen",
                            raw_email=email_content
                        )
                    
                    # Map Risk Assessment (handling variations from normalization)
                    if "risk_score" in quote_data:
                        from src.pipeline.models import RiskAssessment
                        result.risk_assessment = RiskAssessment(
                            overall_risk_level=quote_data.get("risk_level", "UNKNOWN"),
                            risk_score=float(quote_data.get("risk_score", 0)),
                            recommendation=quote_data.get("recommendation", "REFER")
                        )

                    # Map Premium/Quote
                    # Normalization ensures "Total Premium" becomes "total_premium"
                    total_prem = 0.0
                    if "total_premium" in quote_data:
                        total_prem = float(quote_data["total_premium"])
                    elif "premium" in quote_data:
                         total_prem = float(quote_data["premium"])
                         
                    if total_prem > 0:
                        # Modifier Result (for premium field)
                        from src.pipeline.models import ModifierResult
                        result.modifier_result = ModifierResult(
                            adjusted_premium=total_prem, 
                            total_modifier_impact=0.0,
                            total_modifier_percentage=0.0,
                            modifiers_applied=[]
                        )

                        # Generated Quote (for breakdown)
                        from src.pipeline.models import GeneratedQuote, QuotePremiumSummary
                        breakdown = []
                        # Normalization ensures "Premium Breakdown" becomes "premium_breakdown"
                        if "premium_breakdown" in quote_data:
                            for item in quote_data["premium_breakdown"]:
                                breakdown.append(QuotePremiumSummary(
                                    coverage_type=item.get("coverage_type", "General"),
                                    premium=float(item.get("premium", 0)),
                                    limits=item.get("limits")
                                ))
                        
                        result.generated_quote = GeneratedQuote(
                            quote_id=result.quote_id,
                            client_name=quote_data.get("client_name", "Unknown"),
                            quote_valid_until="30 Days",
                            total_annual_premium=total_prem,
                            premium_summary=breakdown,
                            coverage_summary="Generated by AutoGen",
                            quote_letter=quote_content
                        )
                else:
                    logger.warning("Failed to extract JSON from AutoGen output")
            else:
                result.success = False
                result.errors = [autogen_result.get("error", "AutoGen processing failed")]

            self._report_progress(1, "Microsoft AutoGen Pipeline", "complete")

        except Exception as e:
            logger.exception(f"❌ Microsoft AutoGen pipeline error: {e}")
            result.success = False
            result.errors = [str(e)]

        # Record basic metrics
        total_duration = time.time() - start_time
        result.metrics = PipelineMetrics(
            total_duration_seconds=round(total_duration, 2),
            step_durations={"autogen_pipeline": round(total_duration, 2)},
            llm_calls=1,  # Simplified metric
            vector_searches=0,
            documents_retrieved=0,
        )

        # Save to MongoDB regardless of success (to track errors)
        if result.success:
            self._save_result(result)

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
