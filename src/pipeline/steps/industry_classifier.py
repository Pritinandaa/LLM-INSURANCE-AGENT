"""
Step 2: Industry Classifier
Determines the Business Industry Classification (BIC) code for the client.
"""

from src.core.fireworks_client import FireworksClient
from src.core.vector_search import VectorSearchService
from src.core.mongodb_client import Collections
from src.prompts import INDUSTRY_CLASSIFIER_SYSTEM, INDUSTRY_CLASSIFIER_PROMPT
from src.pipeline.models import ExtractedEmail, IndustryClassification


class IndustryClassifierStep:
    """
    Classifies the business into an industry category using BIC codes.
    Uses vector search to find relevant classifications and LLM to select best match.
    """

    def __init__(
        self,
        llm_client: FireworksClient,
        vector_search: VectorSearchService
    ):
        """Initialize with required services."""
        self.llm_client = llm_client
        self.vector_search = vector_search

    def execute(self, extracted_email: ExtractedEmail) -> IndustryClassification:
        """
        Classify the business industry.

        Args:
            extracted_email: Parsed email data

        Returns:
            IndustryClassification with BIC code and details
        """
        # Build search query from business description
        search_query = f"{extracted_email.industry_description} {extracted_email.client_name}"

        # Search for matching BIC codes
        rag_result = self.vector_search.rag_query(
            collection_name=Collections.BIC_CODES,
            query=search_query,
            limit=5,
        )

        # Format context for LLM
        prompt = INDUSTRY_CLASSIFIER_PROMPT.format(
            business_description=extracted_email.industry_description,
            bic_codes_context=rag_result["context"],
        )

        result = self.llm_client.generate_json(
            prompt=prompt,
            system_prompt=INDUSTRY_CLASSIFIER_SYSTEM,
            temperature=0.1,
        )

        return IndustryClassification(
            bic_code=result.get("bic_code", "99"),
            industry_name=result.get("industry_name", "Unknown"),
            risk_category=result.get("risk_category", "MEDIUM"),
            confidence_score=float(result.get("confidence_score", 0.5)),
            matching_keywords=result.get("matching_keywords", []),
            subcategory=result.get("subcategory"),
        )
