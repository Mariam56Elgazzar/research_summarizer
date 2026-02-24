"""
app/services/llm_service.py
Groq API gateway. All LLM communication lives here.
"""
import logging
import time

from groq import Groq, APIConnectionError, APIStatusError, RateLimitError, AuthenticationError

from app.core.config import Settings
from app.core.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


def create_groq_client(api_key: str) -> Groq:
    """Instantiate and return a Groq client."""
    return Groq(api_key=api_key)


def chat_completion(
    client: Groq,
    system_prompt: str,
    user_prompt: str,
    settings: Settings,
    max_tokens: int | None = None,
) -> str:
    """Call Groq chat completion with retry logic.

    Args:
        client: Authenticated Groq client.
        system_prompt: System message content.
        user_prompt: User message content.
        settings: App settings (model, retries, delay).
        max_tokens: Override max tokens (defaults to settings.max_tokens_summary).

    Returns:
        LLM response text as a string.

    Raises:
        LLMServiceError: On auth failure, rate limit, or exhausted retries.
    """
    if max_tokens is None:
        max_tokens = settings.max_tokens_summary

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_exc: Exception | None = None

    for attempt in range(1, settings.llm_max_retries + 1):
        try:
            logger.debug("Groq API call attempt %d/%d", attempt, settings.llm_max_retries)
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            content = response.choices[0].message.content
            logger.debug("Groq response received (%d chars)", len(content or ""))
            return content or ""

        except AuthenticationError as exc:
            raise LLMServiceError(
                "Invalid Groq API key. Please check your credentials.", original=exc
            ) from exc

        except RateLimitError as exc:
            raise LLMServiceError(
                "Groq rate limit reached. Please wait a moment and try again.", original=exc
            ) from exc

        except (APIConnectionError, APIStatusError) as exc:
            last_exc = exc
            logger.warning("Groq API error on attempt %d: %s", attempt, exc)
            if attempt < settings.llm_max_retries:
                time.sleep(settings.llm_retry_delay * attempt)
            else:
                raise LLMServiceError(
                    f"Groq API error: {exc}", original=exc
                ) from exc

        except Exception as exc:
            last_exc = exc
            logger.warning("Unexpected error on attempt %d: %s", attempt, exc)
            if attempt < settings.llm_max_retries:
                time.sleep(settings.llm_retry_delay * attempt)
            else:
                raise LLMServiceError(
                    f"Unexpected error calling Groq: {type(exc).__name__}: {exc}",
                    original=exc,
                ) from exc

    raise LLMServiceError(
        f"Groq API call failed after {settings.llm_max_retries} attempts: {last_exc}",
        original=last_exc,
    )
