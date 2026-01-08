"""
Step 1: Email Parser
Extracts structured information from unstructured broker emails.
"""

from typing import Optional
from src.core.fireworks_client import FireworksClient
from src.prompts import EMAIL_PARSER_SYSTEM, EMAIL_PARSER_PROMPT
from src.pipeline.models import ExtractedEmail, CoverageRequest, BrokerContact


class EmailParserStep:
    """
    Parses broker quote request emails and extracts structured data.
    Uses LLM to understand natural language and extract key fields.
    """

    def __init__(self, llm_client: FireworksClient):
        """Initialize with LLM client."""
        self.llm_client = llm_client

    def execute(self, email_content: str) -> ExtractedEmail:
        """
        Parse email and extract structured information.

        Args:
            email_content: Raw email text

        Returns:
            ExtractedEmail with all extracted fields
        """
        prompt = EMAIL_PARSER_PROMPT.format(email_content=email_content)

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=EMAIL_PARSER_SYSTEM,
            temperature=0.1,
        )

        # Parse coverage requests
        coverages = []
        for cov in result.get("coverage_requested", []) or []:
            if isinstance(cov, dict):
                coverages.append(CoverageRequest(
                    coverage_type=cov.get("coverage_type", "unknown"),
                    limits=cov.get("limits"),
                    additional_details=cov.get("additional_details"),
                ))
            elif isinstance(cov, str):
                coverages.append(CoverageRequest(coverage_type=cov))

        # Parse broker info
        broker_data = result.get("broker", {}) or {}
        broker = None
        if broker_data:
            broker = BrokerContact(
                name=broker_data.get("name"),
                email=broker_data.get("email"),
                phone=broker_data.get("phone"),
                brokerage=broker_data.get("brokerage"),
            )

        return ExtractedEmail(
            client_name=result.get("client_name", "Unknown Client"),
            industry_description=result.get("industry_description", ""),
            location=result.get("location"),
            annual_revenue=self._parse_number(result.get("annual_revenue")),
            employee_count=self._parse_int(result.get("employee_count")),
            years_in_business=self._parse_int(result.get("years_in_business")),
            coverage_requested=coverages,
            vehicle_count=self._parse_int(result.get("vehicle_count")),
            property_value=self._parse_number(result.get("property_value")),
            loss_history=result.get("loss_history"),
            effective_date=result.get("effective_date"),
            urgency=result.get("urgency"),
            broker=broker,
            raw_email=email_content,
        )

    def _parse_number(self, value) -> Optional[float]:
        """Parse a number from various formats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove currency symbols and commas
            cleaned = value.replace("$", "").replace(",", "").strip()
            # Handle M/K suffixes
            if cleaned.upper().endswith("M"):
                return float(cleaned[:-1]) * 1_000_000
            if cleaned.upper().endswith("K"):
                return float(cleaned[:-1]) * 1_000
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _parse_int(self, value) -> Optional[int]:
        """Parse an integer value."""
        num = self._parse_number(value)
        return int(num) if num is not None else None
