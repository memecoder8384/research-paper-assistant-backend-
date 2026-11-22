import json
import logging
import re
from typing import Any, Dict, List, Optional
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def extract_json_from_llm(text: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM responses, stripping any markdown code fences."""
    cleaned = text.strip()
    # Remove markdown code block fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as err:
        logger.warning("Failed direct JSON decode: %s. Attempting bracket extraction.", err)
        # Attempt to isolate outer JSON object { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}...") from err


class ResearchOrchestrator:
    """Central AI orchestrator that coordinates LLM generation, prompting, and structured output parsing."""

    @staticmethod
    def call_structured_llm(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1800,
    ) -> Dict[str, Any]:
        """Execute Groq chat completion and return parsed JSON response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_output = LLMService.generate_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return extract_json_from_llm(raw_output)
