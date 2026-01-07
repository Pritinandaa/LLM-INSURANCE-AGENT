import json
import re
import logging

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> dict:
    """
    Robustly extract the largest JSON object from a string.
    Handles markdown code blocks, raw text, and nested structures.
    """
    if not text:
        return {}
        
    # Attempt 1: Strip Markdown code blocks
    clean_text = text
    if "```" in text:
        # Try to find json block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if match:
            clean_text = match.group(1)
        else:
            # Fallback to simple split if regex fails (e.g. multiple blocks)
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    clean_text = part[4:].strip()
                    break
                if part.startswith("{"):
                    clean_text = part
                    break
    
    # Attempt 2: Find outermost braces {}
    try:
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = clean_text[start_idx : end_idx + 1]
            data = json.loads(json_str)
            return normalize_keys(data)
    except Exception as e:
        logger.warning(f"JSON extraction failed: {e}")
        
    return {}

def normalize_keys(data: dict) -> dict:
    """
    Recursively normalize dictionary keys to snake_case.
    Also flattens 'quote' or 'response' wrappers.
    """
    if not isinstance(data, dict):
        return data
        
    # Flatten wrappers
    if "quote" in data and isinstance(data["quote"], dict):
        return normalize_keys(data["quote"])
    if "response" in data and isinstance(data["response"], dict):
        return normalize_keys(data["response"])
        
    new_data = {}
    for k, v in data.items():
        # Convert "Total Premium" -> "total_premium"
        new_key = k.lower().replace(" ", "_").strip()
        
        # Handle "premium_breakdown" list recursively
        if isinstance(v, list):
            new_data[new_key] = [normalize_keys(item) for item in v if isinstance(item, dict)]
        elif isinstance(v, dict):
            new_data[new_key] = normalize_keys(v)
        else:
            new_data[new_key] = v
            
    return new_data
