import os
import time

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
# Необязательно: OpenAI-совместимый прокси (LiteLLM, OpenRouter, локальный шлюз и т.п.)
ENV_BASE_URL = "OPENAI_BASE_URL"
# Допустимо также OPENAI_API_BASE (часто встречается в примерах прокси)
ENV_BASE_URL_ALT = "OPENAI_API_BASE"
# Имя модели на стороне прокси (если не задано — используется DEFAULT_MODEL)
ENV_MODEL = "OPENAI_MODEL"
ENV_MODEL_ALT = "LLM_MODEL"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
MAX_ARTICLE_CHARS = 12000

SYSTEM_PROMPT = (
    "Ты — профессиональный редактор email-рассылок для деловой аудитории. "
    "Пиши только на русском языке. Стиль письма — экспертный, деловой и ясный. "
    "Не используй HTML и Markdown. Не выдумывай факты, которых нет в статье."
)


def _build_user_prompt(article_text: str) -> str:
    """Build user message with truncated article and strict output requirements."""
    excerpt = article_text[:MAX_ARTICLE_CHARS]
    return f"""Ниже текст статьи (возможно, обрезанный до {MAX_ARTICLE_CHARS} символов). На его основе подготовь одну email-рассылку.

--- Начало статьи ---
{excerpt}
--- Конец статьи ---

Верни ответ строго в таком формате (без HTML и Markdown, без пояснений до или после блока):

Тема: ...
Preheader: ...

Тело:
Здравствуйте!

[Основной текст письма]

CTA: Позвоните менеджеру нашей компании, чтобы организовать встречу и обсудить, как применить эти идеи в вашей ситуации.

Требования к письму:
- язык ответа — всегда русский;
- строка «Тема:» — до 80 символов (считай только текст темы после двоеточия и пробела);
- строка «Preheader:» — до 120 символов (только текст preheader после двоеточия и пробела);
- блок «Тело:» — от 900 до 1500 символов целиком (включая «Здравствуйте!», основной текст и строку CTA как указано выше);
- экспертный и деловой стиль;
- без кликбейта;
- без чрезмерно рекламного тона;
- без эмоционального давления;
- не копируй статью целиком — перескажи суть своими словами;
- не добавляй HTML, Markdown, списки разметки, заголовки разметки;
- не добавляй технические комментарии, примечания редактора или мета-текст;
- не выдумывай факты, цифры, имена, выводы или обещания — опирайся только на статью;
- избегай повторов;
- сделай текст пригодным для реальной email-рассылки."""


def _resolve_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "API key is not set. Set OPENAI_API_KEY (or API_KEY for совместимости) "
            "in the environment or in a .env file."
        )
    return key


def _resolve_base_url() -> str | None:
    raw = (os.environ.get(ENV_BASE_URL) or os.environ.get(ENV_BASE_URL_ALT) or "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


def _resolve_model_name() -> str:
    name = (os.environ.get(ENV_MODEL) or os.environ.get(ENV_MODEL_ALT) or "").strip()
    return name or DEFAULT_MODEL


def _get_client() -> OpenAI:
    """Return OpenAI SDK client (официальный API или OpenAI-совместимый прокси по base_url)."""
    api_key = _resolve_api_key()
    base_url = _resolve_base_url()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        return code is not None and (code == 429 or code >= 500)
    return False


def generate_email(article_text: str) -> str:
    """Generate newsletter text from article content via OpenAI Chat Completions."""
    if not isinstance(article_text, str):
        raise ValueError("article_text must be a non-empty string.")
    if not article_text.strip():
        raise ValueError("article_text must be a non-empty string.")

    client = _get_client()
    model = _resolve_model_name()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(article_text)},
    ]

    last_error: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
            )
            choice = response.choices[0] if response.choices else None
            content = choice.message.content if choice else None
            if content is None or not str(content).strip():
                raise RuntimeError("OpenAI returned an empty completion.")
            return str(content).strip()
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(RETRY_DELAY_SECONDS * attempt)

    raise RuntimeError(
        f"OpenAI request failed after {MAX_RETRIES} attempts."
    ) from last_error
