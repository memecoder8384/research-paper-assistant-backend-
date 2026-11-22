import logging
import threading
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Thread-safe service for generating chat completions using the Google Gemini API."""

    _client: Optional[Any] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> genai.Client:
        """Initialize or reuse the Gemini API client safely."""
        if cls._client is not None:
            return cls._client

        api_key = settings.gemini_api_key or settings.google_api_key
        if not api_key or not api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set your GEMINI_API_KEY in the environment or .env file."
            )

        with cls._lock:
            if cls._client is None:
                cls._client = genai.Client(api_key=api_key.strip())
        return cls._client

    @classmethod
    def set_client(cls, client: Any) -> None:
        """Inject a custom or mock Gemini client (useful for unit tests without API calls)."""
        with cls._lock:
            cls._client = client

    @classmethod
    def reset_client(cls) -> None:
        """Reset the cached client instance."""
        with cls._lock:
            cls._client = None

    @classmethod
    def generate_chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call Gemini API with error and rate limit handling.

        - Uses configurable model, temperature, and token limit
        - Never exposes credentials in exceptions or logs
        - Maps external Gemini errors to clean application exceptions
        """
        client = cls.get_client()
        target_model = model or settings.gemini_model
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        system_instruction = None
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            else:
                user_parts.append(f"[{role.upper()}]: {content}")

        prompt_content = "\n\n".join(user_parts) if user_parts else (system_instruction or "")

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temp,
            max_output_tokens=tokens,
        )

        try:
            # Handle mock client or official genai.Client
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                response = client.models.generate_content(
                    model=target_model,
                    contents=prompt_content,
                    config=config,
                )
            elif hasattr(client, "chat") and hasattr(client.chat, "completions"):
                # Backward compatibility for legacy test mocks
                resp = client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                return resp.choices[0].message.content.strip()
            else:
                raise RuntimeError("Unsupported client configured for LLMService.")

            if not response or not response.text:
                raise RuntimeError("Empty response received from Gemini API.")
            return response.text.strip()

        except APIError as api_err:
            err_msg = str(api_err)
            if "RESOURCE_EXHAUSTED" in err_msg or "rate limit" in err_msg.lower() or "429" in err_msg:
                logger.error("Gemini rate limit exceeded: %s", api_err)
                raise RuntimeError("Gemini API rate limit exceeded. Please wait a moment and try again.") from api_err
            elif "INVALID_ARGUMENT" in err_msg or "API_KEY_INVALID" in err_msg or "401" in err_msg or "403" in err_msg:
                logger.error("Gemini authentication failed: Invalid API key")
                raise ValueError("Invalid Gemini API key. Please check your credentials.") from api_err
            elif "DEADLINE_EXCEEDED" in err_msg or "timeout" in err_msg.lower():
                logger.error("Gemini API request timed out: %s", api_err)
                raise RuntimeError("Gemini API request timed out. Please try again.") from api_err
            else:
                logger.error("Gemini API error: %s", api_err)
                raise RuntimeError(f"Gemini API error: {err_msg}") from api_err
        except Exception as exc:
            exc_name = type(exc).__name__
            if "RateLimit" in exc_name or "429" in str(exc):
                logger.error("Gemini rate limit exceeded: %s", exc)
                raise RuntimeError("Gemini API rate limit exceeded. Please wait a moment and try again.") from exc
            if "Timeout" in exc_name or "Deadline" in exc_name:
                logger.error("Gemini request timed out: %s", exc)
                raise RuntimeError("Gemini API request timed out. Please try again.") from exc
            if isinstance(exc, (ValueError, RuntimeError)):
                raise
            logger.error("Gemini request failed: %s", exc)
            raise RuntimeError(f"Gemini API request failed: {str(exc)}") from exc
