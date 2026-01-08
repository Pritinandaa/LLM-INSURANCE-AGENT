import json
import logging
from typing import Annotated
from semantic_kernel.functions import kernel_function
from src.prompts import EMAIL_PARSER_SYSTEM, EMAIL_PARSER_PROMPT, RISK_ASSESSMENT_SYSTEM, RISK_ASSESSMENT_PROMPT
from src.core.mongodb_client import get_collection, Collections
from datetime import datetime

logger = logging.getLogger(__name__)

class UnderwritingPlugins:
    @kernel_function(
        description="Parses a broker email and extracts structured underwriting data.",
        name="parse_email"
    )
    async def parse_email(
        self,
        email_content: Annotated[str, "The raw content of the broker email"]
    ) -> str:
        """
        Extracts structured information from the email content.
        """
        # Note: In a real SK setup, the kernel would call the LLM using the prompt.
        # Here we are defining the function that can be used by an agent or the kernel.
        # However, for agents to 'work together', they often use plugins to perform specific tasks.
        
        # We'll return the prompt that the agent should use or handle the extraction here.
        # To make it truly agentic, the EmailAgent will use this plugin.
        
        # For simplicity in this agentic flow, we'll let the agent handle the prompt 
        # but this plugin can provide the schema or post-processing.
        return f"Please extract data from this email: {email_content}"

    @kernel_function(
        description="Evaluates the risk of an insurance application and provides a recommendation.",
        name="evaluate_risk"
    )
    def evaluate_risk(
        self,
        client_data: Annotated[str, "The extracted structured data of the client"]
    ) -> str:
        """
        Analyzes the risk based on the provided client data.
        """
        return f"Evaluate the risk for this client: {client_data}"

class DatabasePlugin:
    @kernel_function(
        description="Saves the underwriting result to MongoDB.",
        name="save_result"
    )
    def save_result(
        self,
        result_json: Annotated[str, "The final underwriting result in JSON format"]
    ) -> str:
        """
        Saves the processed quote/risk assessment to the database.
        """
        try:
            data = json.loads(result_json)
            collection = get_collection(Collections.QUOTES)
            data["saved_at"] = datetime.utcnow()
            if "quote_id" not in data:
                data["quote_id"] = f"Q-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            collection.replace_one(
                {"quote_id": data["quote_id"]},
                data,
                upsert=True
            )
            return f"Successfully saved result with ID: {data['quote_id']}"
        except Exception as e:
            logger.error(f"Failed to save to MongoDB: {e}")
            return f"Error saving to database: {str(e)}"
