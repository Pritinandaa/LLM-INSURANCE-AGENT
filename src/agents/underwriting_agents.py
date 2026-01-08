import os
import json
import asyncio
import logging
from typing import List, Optional
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import AuthorRole

from src.core.semantic_kernel_setup import get_kernel
from src.agents.underwriting_plugins import UnderwritingPlugins, DatabasePlugin
from src.prompts import (
    EMAIL_PARSER_SYSTEM, 
    EMAIL_PARSER_PROMPT,
    RISK_ASSESSMENT_SYSTEM,
    RISK_ASSESSMENT_PROMPT,
)

# User's suggested imports for settings
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings

logger = logging.getLogger(__name__)

def _strip_json(text: str) -> str:
    """Helper to remove markdown backticks from a JSON string."""
    if not text: return "{}"
    t = text.strip()
    if t.startswith("```"):
        # Remove starting ```json or ```
        if t.startswith("```json"):
            t = t[7:]
        else:
            t = t[3:]
        # Remove ending ```
        if t.endswith("```"):
            t = t[:-3]
    return t.strip() or "{}"

def _to_text(msg) -> str:
    """Helper to safely extract string content from a message object."""
    if not msg: return ""
    if isinstance(msg, str): return msg
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content: return content
    items = getattr(msg, "items", [])
    if items:
        parts = []
        for it in items:
            if hasattr(it, "text") and it.text:
                parts.append(str(it.text))
            elif isinstance(it, str):
                parts.append(it)
        if parts: return "\n".join(parts).strip()
    return str(content) if content else ""

class UnderwritingAgentOrchestrator:
    def __init__(self):
        # We use a clean kernel for agents to avoid 400 errors with NVIDIA NIM
        self.agent_kernel = get_kernel()

    def _create_agents(self):
        """Build PRO-LEVEL agents with full expert logic for NVIDIA NIM."""
        
        # 1. Broker Agent - High-precision Data Extraction
        extract_schema = """
Extract following JSON structure:
{
  "client_name": "Entity name",
  "industry_description": "Sector/SIC/BIC info",
  "location": "State/Region",
  "annual_revenue": 1000000,
  "employee_count": 50,
  "loss_history": "Detailed history or 'None mentioned'",
  "coverage_requested": "Limits and types",
  "urgency": "High/Normal"
}
"""
        broker_instr = (
            f"{EMAIL_PARSER_SYSTEM}\n\n"
            "TASK: Extract underwriting data based on the JSON schema below.\n"
            f"{extract_schema}\n"
            "STRICT RULES:\n"
            "1. Analyze the text between <email> tags only.\n"
            "2. Return ONLY raw JSON (no markdown boxes if possible, but keep it valid).\n"
            "3. If a field is missing, use null or 'Unknown'.\n"
            "4. DO NOT include any introductory or concluding text."
        )
        
        broker_agent = ChatCompletionAgent(
            kernel=self.agent_kernel,
            name="BrokerAgent",
            prompt_template_config=PromptTemplateConfig(
                template=broker_instr,
                allow_dangerously_set_content=True
            )
        )

        # 2. Underwriter Agent - Senior Risk Assessment
        underwriter_instr = (
            f"{RISK_ASSESSMENT_SYSTEM}\n\n"
            "TASK: Evaluate the risk based ON THE EXTRACTED DATA PROVIDED.\n\n"
            "STRICT ANTI-HALLUCINATION RULES:\n"
            "1. ONLY use data found between <extracted_data> tags.\n"
            "2. DO NOT make up hypothetical scenarios if data is missing.\n"
            "3. If critical data (Industry, Revenue) is 'Unknown', your verdict MUST be 'MANUAL REVIEW'.\n"
            "4. Identify the BIC/SIC risk accurately based on the description.\n\n"
            "STRUCTURE YOUR RESPONSE AS:\n"
            "ANALYSIS BASIS: [Specific reasons from data]\n"
            "RISK LEVEL: [LOW/MEDIUM/HIGH/VERY_HIGH]\n"
            "VERDICT: [SAFE, NOT SAFE, or MANUAL REVIEW]"
        )

        underwriter_agent = ChatCompletionAgent(
            kernel=self.agent_kernel,
            name="UnderwriterAgent",
            prompt_template_config=PromptTemplateConfig(
                template=underwriter_instr,
                allow_dangerously_set_content=True
            )
        )
        return broker_agent, underwriter_agent

    async def run_workflow(self, email_content: str):
        email_text = (email_content or "").strip()
        broker_agent, underwriter_agent = self._create_agents()

        exec_settings = OpenAIChatPromptExecutionSettings(
            temperature=0.0, # Purely deterministic for "Real" details
            max_tokens=1024,
            function_choice_behavior=None
        )

        # -------- Step 1: Real Extraction --------
        broker_history = ChatHistory()
        broker_history.add_message(ChatMessageContent(
            role=AuthorRole.USER, 
            content=f"EXTRACT FROM EMAIL CONTENT BELOW:\n<email>\n{email_text}\n</email>"
        ))

        print(f"[WORKFLOW] Calling Llama Data Extraction Engine...")
        broker_response = await broker_agent.get_response(
            chat_history=broker_history,
            settings=exec_settings
        )
        broker_text = _strip_json(_to_text(broker_response))

        # -------- Step 2: Real Underwriting --------
        underwriter_history = ChatHistory()
        underwriter_history.add_message(ChatMessageContent(
            role=AuthorRole.USER, 
            content=(
                "ANALYZE RISK BASED ON THESE INPUTS:\n\n"
                f"<source_email>\n{email_text}\n</source_email>\n\n"
                f"<extracted_data>\n{broker_text}\n</extracted_data>\n\n"
                "Provide your professional BASIS, RISK LEVEL, and VERDICT."
            )
        ))
        
        print(f"[WORKFLOW] Calling Llama Risk Decision Engine...")
        underwriter_response = await underwriter_agent.get_response(
            chat_history=underwriter_history,
            settings=exec_settings
        )
        underwriter_text = _to_text(underwriter_response)

        # -------- Result --------
        return [
            {"agent": "BrokerAgent", "content": broker_text},
            {"agent": "UnderwriterAgent", "content": underwriter_text},
            {"agent": "System", "content": "Underwriting completed by NVIDIA Llama Engines."},
        ]

def process_underwriting_request(email_content: str):
    return asyncio.run(UnderwritingAgentOrchestrator().run_workflow(email_content))
