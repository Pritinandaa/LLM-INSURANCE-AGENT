import os
import logging
import warnings
from dotenv import load_dotenv
import semantic_kernel as sk
from openai import AsyncOpenAI
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# Suppress ALL Vertex AI warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

def get_kernel() -> sk.Kernel:
    """
    Initialize and return a Semantic Kernel instance.
    STRICTLY FORCES NVIDIA NIM using the user's specific AsyncOpenAI pattern.
    """
    kernel = sk.Kernel()

    # Log current config status
    # Note: User snippet uses NVIDIA_NIM_API_KEY
    nvidia_key = os.getenv("NVIDIA_NIM_API_KEY", "").strip()
    nim_model = os.getenv("NVIDIA_CHAT_MODEL", "meta/llama-3.3-70b-instruct")
    
    print(f"\n[AI-INIT] Using NVIDIA NIM...")
    
    if not nvidia_key or nvidia_key == "API_KEY" or not nvidia_key.startswith("nvapi-"):
        print(f"[AI-INIT] ERROR: NVIDIA_NIM_API_KEY is missing or invalid in .env")
        raise ValueError("NVIDIA_NIM_API_KEY is required to run the agents with Llama.")

    try:
        print(f"[AI-INIT] >>> CONNECTING TO NVIDIA NIM (Model: {nim_model})")
        
        # Exact user pattern for AsyncOpenAI
        nim_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key
        )
        
        openai_service = OpenAIChatCompletion(
            ai_model_id=nim_model,
            async_client=nim_client
        )
        kernel.add_service(openai_service)
        return kernel
        
    except Exception as e:
        print(f"[AI-INIT] !!! INITIALIZATION FAILED: {str(e)}")
        raise RuntimeError(f"NVIDIA NIM failed to initialize. Error: {e}")
