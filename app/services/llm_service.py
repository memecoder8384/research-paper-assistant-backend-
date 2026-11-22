import logging
import threading
from typing import Any, Dict, List, Optional
from groq import APIConnectionError, APITimeoutError, AuthenticationError, Groq, GroqError, RateLimitError
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Thread-safe service for generating chat completions using the Groq API."""

    _client: Optional[Any] = None
    _lock = threading.Lock()

    @classmethod
    def get_client(cls) -> Groq:
        """Initialize or reuse the Groq API client safely."""
        if cls._client is not None:
            return cls._client

        api_key = settings.groq_api_key
        if not api_key or not api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set your GROQ_API_KEY in the environment or .env file."
            )

        with cls._lock:
            if cls._client is None:
                cls._client = Groq(api_key=api_key.strip())
        return cls._client

    @classmethod
    def set_client(cls, client: Any) -> None:
        """Inject a custom or mock Groq client (useful for unit tests without API calls)."""
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
        """Call Groq chat completions API with error and rate limit handling.

        - Uses configurable model, temperature, and token limit
        - Never exposes credentials in exceptions or logs
        - Maps external Groq errors to clean application exceptions
        """
        client = cls.get_client()
        target_model = model or settings.groq_model
        temp = temperature if temperature is not None else settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens

        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            return content.strip()

        except AuthenticationError as auth_err:
            logger.error("Groq authentication failed: Invalid API key")
            raise ValueError("Invalid Groq API key. Please check your credentials.") from auth_err

        except RateLimitError as rate_err:
            logger.error("Groq rate limit exceeded: %s", rate_err)
            raise RuntimeError("Groq API rate limit exceeded. Please wait a moment and try again.") from rate_err

        except APITimeoutError as timeout_err:
            logger.error("Groq API request timed out: %s", timeout_err)
            raise RuntimeError("Groq API request timed out. Please try again.") from timeout_err

        except APIConnectionError as conn_err:
            logger.error("Failed to connect to Groq API: %s", conn_err)
            raise RuntimeError("Unable to connect to Groq API. Please check your network connection.") from conn_err

        except GroqError as groq_err:
            logger.error("Groq API error: %s", groq_err)
            raise RuntimeError(f"Groq API error: {str(groq_err)}") from groq_err

        except Exception as exc:
            logger.error("Unexpected error during LLM generation: %s", exc)
            raise RuntimeError(f"LLM generation failed: {str(exc)}") from exc
