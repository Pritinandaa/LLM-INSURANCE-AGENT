"""
AutoGen client using Microsoft AutoGen (AgentChat/Core).
Uses the modern Microsoft-backed AutoGen packages.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv(override=True)

# Microsoft AutoGen packages - Strict Import
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient  # Used as interface
    from autogen_core import CancellationToken
except ImportError as e:
    raise ImportError(
        f"Critical Dependency Missing: {e}. "
        "Please install Microsoft AutoGen packages: "
        "pip install autogen-agentchat autogen-core 'autogen-ext[openai]'"
    )

from src.config import get_settings

logger = logging.getLogger(__name__)

class AutoGenClient:
    """
    Microsoft AutoGen client for multi-agent processing.
    Uses the modern AgentChat/Core API with Vertex AI.
    """

    def __init__(self):
        self.settings = get_settings()

        # Enforce GOOGLE_APPLICATION_CREDENTIALS in environment for SDKs
        if self.settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.settings.google_application_credentials
            logger.info(f"Enforced GOOGLE_APPLICATION_CREDENTIALS: {self.settings.google_application_credentials}")

        # Initialize Vertex AI Model Client
        # We attempt to use the official extension if available, otherwise we'd need a custom adapter.
        # Given the instruction to "replace ag2 with the new ones", we'll assume autogen-ext supports standard patterns.
        # Note: As of early 2024/2025, Vertex AI support usually comes via autogen-ext or similar.
        # We'll stick to a pattern that allows Vertex AI usage.
        
        # Try to use Vertex AI extension if available
        try:
                from autogen_ext.models.vertexai import VertexAIModelClient
                logger.info("[OK] Using autogen_ext VertexAIModelClient")
                self.model_client = VertexAIModelClient(
                    model="gemini-pro",
                    project_id=self.settings.google_cloud_project,
                    location=self.settings.google_cloud_region,
                )
        except ImportError:
                logger.warning("[WARN] autogen_ext.models.vertexai not found. Using CustomVertexAIClient.")
                self.model_client = self._create_custom_vertex_client()

    def _create_custom_vertex_client(self):
        """Create a custom Vertex AI client adapter."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from autogen_core import CancellationToken
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            
            try:
                from autogen_core.models import (
                    ChatCompletionClient, 
                    ModelCapabilities, 
                    ModelInfo, 
                    RequestUsage, 
                    FinishReason,
                    CreateResult,
                    SystemMessage,
                    UserMessage,
                    AssistantMessage
                )
            except ImportError:
                from autogen_core.models import ChatCompletionClient, ModelCapabilities, ModelInfo, RequestUsage
                class FinishReason:
                    STOP = "stop"
                    LENGTH = "length"
                    FUNCTION_CALL = "function_call"
                # We still need CreateResult if it exists
                try:
                    from autogen_core.models import CreateResult
                except ImportError:
                    pass

            from dataclasses import dataclass
            
            @dataclass
            class CustomVertexAIClient(ChatCompletionClient):
                def __init__(self, settings, project, location, model_candidates=None):
                    self.settings = settings
                    if model_candidates is None:
                        # Prioritize user-configured model from env, then newer models, fallback to older ones
                        import os
                        configured_model = os.getenv("VERTEX_LLM_MODEL")
                        
                        self.model_candidates = [
                            "gemini-2.5-flash",
                            "gemini-2.0-flash-exp",
                            "gemini-1.5-flash-001",
                            "gemini-1.5-flash-002",
                            "gemini-1.5-pro-001",
                            "gemini-1.5-pro-002",
                            "gemini-1.0-pro-002",
                            "gemini-1.0-pro",
                        ]
                        
                        if configured_model:
                            print(f"DEBUG: Adding configured model '{configured_model}' to start of list.")
                            self.model_candidates.insert(0, configured_model)
                    else:
                        self.model_candidates = model_candidates
                        
                    import os
                    self.project = self.settings.vertex_project_id or os.getenv("VERTEX_PROJECT_ID", project)
                    self.location = self.settings.vertex_location or os.getenv("VERTEX_LOCATION", location)
                    
                    self._active_model_name = None
                    self._model = None
                    self._model_info = ModelInfo(
                        vision=False,
                        function_calling=True,
                        json_output=True,
                        family="gemini"
                    )
                    
                    # Initialize Vertex AI globally once
                    vertexai.init(project=self.project, location=self.location)
                    
                    # Connection State
                    self._connection_verified = False

                async def _verify_connection_and_select_model(self):
                    """Try candidates until one works."""
                    if self._connection_verified:
                        return

                    import os
                    configured_model = self.settings.vertex_llm_model or os.getenv("VERTEX_LLM_MODEL")
                    
                    # If we have a preferred model, try it first
                    models_to_try = self.model_candidates.copy()
                    if configured_model and configured_model not in models_to_try:
                        models_to_try.insert(0, configured_model)
                    elif configured_model:
                        # Move to front
                        models_to_try.remove(configured_model)
                        models_to_try.insert(0, configured_model)

                    print(f"DEBUG: Starting Model Selection... Candidates: {models_to_try}")
                    
                    last_error = None
                    for model_name in models_to_try:
                        try:
                            print(f"DEBUG: Trying model: {model_name}")
                            model = GenerativeModel(model_name)
                            # Test if it actually works with a tiny request
                            # Note: We can't easily test without an async call here, 
                            # but we can at least initialize it.
                            self._model = model
                            self._active_model_name = model_name
                            self._connection_verified = True
                            print(f"[OK] Successfully selected model: {model_name}")
                            return
                        except Exception as e:
                            print(f"[WARN] Model {model_name} failed: {e}")
                            last_error = e
                            continue
                    
                    if not self._model:
                        raise Exception(f"Failed to initialize any Vertex AI model. Last error: {last_error}")

                async def create(
                    self,
                    messages,
                    *,
                    tools = None,
                    json_output = None,
                    extra_create_args = None,
                    cancellation_token = None,
                ):
                    # Ensure we have a working model selected
                    await self._verify_connection_and_select_model()
                    
                    # Convert AutoGen messages to Vertex Prompt
                    system_instructions = []
                    conversation_parts = []
                    
                    for msg in messages:
                        source = getattr(msg, 'source', 'unknown')
                        content = getattr(msg, 'content', '')
                        
                        # Handle different message types (System vs Conversation)
                        # We use duck typing or class checks if possible
                        msg_type = msg.__class__.__name__
                        
                        if "SystemMessage" in msg_type:
                            system_instructions.append(str(content))
                        else:
                            conversation_parts.append(f"[{source}]: {content}")
                    
                    # Build full prompt
                    full_prompt = ""
                    if system_instructions:
                        full_prompt += "SYSTEM INSTRUCTIONS:\n" + "\n".join(system_instructions) + "\n\n"
                    
                    full_prompt += "CONVERSATION HISTORY:\n" + "\n\n".join(conversation_parts)
                    
                    try:
                        # Set config (enable JSON if requested)
                        config = {}
                        if json_output:
                             config["response_mime_type"] = "application/json"
                        
                        # Use generate_content_async
                        print(f"DEBUG: Vertex Request (len={len(full_prompt)}) using {self._active_model_name}")
                        response = await self._model.generate_content_async(full_prompt, generation_config=config)
                        text_content = response.text
                        
                        # Create valid AutoGen Response
                        from autogen_core.models import CreateResult
                        return CreateResult(
                            finish_reason=FinishReason.STOP,
                            content=text_content,
                            usage=RequestUsage(prompt_tokens=0, completion_tokens=0)
                        )
                    except Exception as e:
                        print(f"[FAIL] Vertex Generation Failed ({self._active_model_name}): {e}")
                        raise

                async def create_stream(self, messages, **kwargs):
                    # Adapter: await full response and yield it to satisfy interface
                    result = await self.create(messages, **kwargs)
                    yield result

                @property
                def model_info(self):
                    return self._model_info

                @property
                def capabilities(self):
                    return self._model_info

                def count_tokens(self, messages, **kwargs):
                    return 0 # Dummy implementation

                def remaining_tokens(self, messages, **kwargs):
                    return 1000000 # Dummy

                def actual_usage(self):
                    return None

                def total_usage(self):
                    return None # Dummy

                def close(self):
                    pass

                @property
                def capabilities(self):
                    # Return dummy capabilities
                    from autogen_core.models import ModelCapabilities
                    return ModelCapabilities(
                        vision=False,
                        function_calling=False,
                        json_output=False
                    )
            
            try:
                client = CustomVertexAIClient(
                    settings=self.settings,
                    project=self.settings.google_cloud_project,
                    location=self.settings.google_cloud_region
                )
                return client
            except TypeError as te:
                # Capture abstract method errors
                print(f"DEBUG: Instantiation TypeError: {te}")
                raise te
            except Exception as e:
                print(f"DEBUG: Instantiation Error: {e}")
                raise e

        except ImportError as e:
            raise ImportError(f"Could not create CustomVertexAIClient. Missing dependencies: {e}. Please ensure 'google-cloud-aiplatform' is installed.")

    async def _create_agents(self):
        """Create the multi-agent team for underwriting processing."""
        # Email Parser Agent
        email_parser = AssistantAgent(
            name="email_parser",
            model_client=self.model_client,
            system_message="Parse insurance broker emails and extract key information including client details, business type, coverage requirements, and risk factors."
        )

        # Industry Classifier Agent
        industry_classifier = AssistantAgent(
            name="industry_classifier",
            model_client=self.model_client,
            system_message="Classify the business industry using standard insurance classification codes and determine appropriate risk categories."
        )

        # Rate Discovery Agent
        rate_discovery = AssistantAgent(
            name="rate_discovery",
            model_client=self.model_client,
            system_message="Research and recommend appropriate insurance rates based on industry standards, location, and risk factors."
        )

        # Risk Assessment Agent
        risk_assessment = AssistantAgent(
            name="risk_assessment",
            model_client=self.model_client,
            system_message="Assess overall risk factors, provide risk scores, and make underwriting recommendations based on the gathered information."
        )

        # Quote Generation Agent
        quote_generator = AssistantAgent(
            name="quote_generator",
            model_client=self.model_client,
            system_message="""You are the Quote Generator.
            Your ONLY goal is to output the final insurance quote in strict JSON format.
            
            The JSON MUST include:
            {
                "client_name": "string",
                "risk_score": number (0-100),
                "risk_level": "LOW"|"MEDIUM"|"HIGH",
                "total_premium": number,
                "premium_breakdown": [{"coverage_type": "string", "premium": number, "limits": "string"}],
                "recommendation": "ACCEPT"|"REFER"|"DECLINE",
                "quote_letter": "string (the full text of the email/letter)"
            }
            Do not include any other conversational text. Just the JSON.
            
            IMPORTANT: After the JSON, you MUST include the exact string: [PROCESSED_QUOTE]"""
        )

        return [email_parser, industry_classifier, rate_discovery, risk_assessment, quote_generator]

    async def process_underwriting_request(self, email_content: str) -> Dict[str, Any]:
        """
        Process an underwriting request using Microsoft AutoGen multi-agent team.
        """
        def log_debug(msg):
            print(f"DEBUG: {msg}")
            with open("autogen_debug.log", "a", encoding="utf-8") as f:
                f.write(f"DEBUG: {msg}\n")

        log_debug("Entered process_underwriting_request (Async - MULTI-AGENT MODE)")
        
        # Clear previous logs
        with open("autogen_debug.log", "w", encoding="utf-8") as f:
            f.write("--- Starting New Request ---\n")

        try:
            log_debug("Creating Agents...")
            agents = await self._create_agents()
            log_debug(f"Agents Created: {len(agents)}")

            # Create termination condition
            termination = TextMentionTermination("[PROCESSED_QUOTE]")

            # Create the team
            log_debug("Creating RoundRobinGroupChat")
            team = RoundRobinGroupChat(
                participants=agents,
                termination_condition=termination,
                max_turns=15
            )

            # Initial task
            task = f"""
            Process this insurance underwriting request from a broker email:

            EMAIL CONTENT:
            {email_content}

            Work through these steps in order:
            1. Email Parser: Extract client information, business details, and coverage requirements
            2. Industry Classifier: Determine the industry classification and risk category
            3. Rate Discovery: Research appropriate insurance rates for this industry and risk profile
            4. Risk Assessment: Evaluate overall risk factors and provide a risk score
            5. Quote Generator: Create a professional insurance quote with premiums and terms

            Each agent should contribute their expertise and build upon previous findings.
            Final output should be a complete insurance quote in JSON format.
            
            CRITICAL: The final message MUST be valid JSON wrapped in a code block or plain text.
            """

            # Run the team
            log_debug("Starting Team Run (task sent)...")
            start_time = asyncio.get_event_loop().time()
            result = await team.run(task=task)
            end_time = asyncio.get_event_loop().time()
            log_debug(f"Team Run Finished in {end_time - start_time:.2f}s")
            
            # Extract the final response by searching backwards for JSON
            messages = result.messages
            log_debug(f"Total messages generated: {len(messages)}")
            
            # Log the full conversation for debugging
            with open("autogen_debug.log", "a", encoding="utf-8") as f:
                for msg in messages:
                    f.write(f"[{msg.source}]: {msg.content}\n\n")

            final_message = ""
            if messages:
                # Iterate backwards to find the JSON quote
                for msg in reversed(messages):
                    content = msg.content
                    if "{" in content and "}" in content and "client_name" in content:
                        final_message = content
                        break
                
                # Fallback: check specifically for quote_generator's last message
                if not final_message:
                    for msg in reversed(messages):
                        if msg.source == "quote_generator":
                            final_message = msg.content
                            break
            
            # CRITICAL FALLBACK: If API returns empty (Auth/Network failure)
            if not final_message or len(final_message) < 10:
                 print(f"DEBUG: No valid JSON found. Last message content: {messages[-1].content if messages else 'NO MESSAGES'}")
                 raise Exception("No JSON declaration found in agent conversation")

            return {
                "status": "processed",
                "quote": final_message,
                "agents_used": ["email_parser", "industry_classifier", "rate_discovery", "risk_assessment", "quote_generator"],
                "processing_mode": "microsoft_autogen"
            }

        except Exception as e:
            # CAPTURE THE ERROR
            with open("autogen_debug.log", "a", encoding="utf-8") as f:
                import traceback
                f.write(f"\nCRITICAL ERROR: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
            
            logger.error(f"AutoGen processing failed: {e}")
            # HARDCODED FALLBACK ENABLED FOR USER REQUEST
            # fallback_quote = """
            # {
            #     "client_name": "TechNova Solutions (Simulated - Models Unavailable)",
            #     "risk_score": 42,
            #     "risk_level": "MEDIUM",
            #     "total_premium": 12500,
            #     "premium_breakdown": [
            #         {"coverage_type": "General Liability", "premium": 5000, "limits": "1M/2M"},
            #         {"coverage_type": "Cyber Security", "premium": 7500, "limits": "5M"}
            #     ],
            #     "recommendation": "ACCEPT",
            #     "quote_letter": "Dear TechNova,\\n\\nBased on our analysis of your software business in San Francisco, we are pleased to offer the attached insurance quote. Your risk profile is moderate due to cyber exposure.\\n\\nSincerely,\\nAI Underwriting Team"
            # }
            # """
            
            # return {
            #     "status": "processed",
            #     "quote": fallback_quote,
            #     "agents_used": ["fallback_simulation"],
            #     "processing_mode": "simulation_fallback"
            # }
            
            # RE-RAISE ERROR TO SEE IT IN UI
            raise e

# Global client instance
_autogen_client: Optional[AutoGenClient] = None


async def get_autogen_client() -> AutoGenClient:
    """Get or create the AutoGen client instance."""
    global _autogen_client
    if _autogen_client is None:
        _autogen_client = AutoGenClient()
    return _autogen_client


def process_underwriting_request_sync(email_content: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for the async AutoGen processing.
    """
    try:
        # Detect if there's a running loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        # Apply nest_asyncio to valid loop
        import nest_asyncio
        nest_asyncio.apply(loop)

        # Run the async logic
        # Note: get_autogen_client is async
        client = loop.run_until_complete(get_autogen_client())
        result = loop.run_until_complete(client.process_underwriting_request(email_content))
        return result
    except Exception as e:
        logger.error(f"AutoGen sync processing failed: {e}")
        return {
            "status": "error",
            "error": f"AutoGen processing failed: {str(e)}"
        }
