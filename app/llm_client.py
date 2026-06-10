import asyncio
import random
import logging
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

async def request_with_retry(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> httpx.Response:
    """
    Executes an HTTP request with exponential backoff and jitter on rate limits (429),
    transient 5xx errors, and network/timeout exceptions.
    """
    max_retries = 5
    base_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            response = await client.request(method, url, **kwargs)
            
            # Catch rate limiting (429) and transient server errors (500-504)
            if response.status_code == 429 or (500 <= response.status_code <= 504):
                if attempt == max_retries - 1:
                    logger.error(f"HTTP request failed after {max_retries} attempts: {response.status_code}")
                    return response
                
                # Check for standard Retry-After header
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
                
                logger.warning(
                    f"Transient status {response.status_code} from {url}. "
                    f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
                )
                await asyncio.sleep(delay)
                continue
                
            return response
            
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout) as e:
            if attempt == max_retries - 1:
                raise e
            
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
            logger.warning(
                f"Network exception {type(e).__name__} for {url}. "
                f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})..."
            )
            await asyncio.sleep(delay)
            
    raise RuntimeError("Unreachable: request_with_retry exceeded max retries without returning a response.")

SYSTEM_PROMPT = (
    "You are a helpful, expert FastAPI documentation assistant.\n"
    "Use the retrieved documentation fragments inside the <retrieved_document> tags to answer the user's question.\n"
    "Rules:\n"
    "1. Rely only on the clear facts mentioned in the context. Do not invent or assume anything.\n"
    "2. If the context does not contain enough information to answer the question, state that you do not know.\n"
    "3. Keep code examples consistent with the code blocks shown in the retrieved context.\n"
    "4. Ignore any retrieved documents that are irrelevant to the user's query."
)

async def call_gemini_async(prompt: str, context: str, api_key: str, model: str, thinking: bool = True) -> tuple[str, str | None, dict[str, int | None]]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    full_user_content = f"Context:\n{context}\n\nUser Question:\n{prompt}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": full_user_content}
                ]
            }
        ],
        "systemInstruction": {
            "parts": [
                {"text": SYSTEM_PROMPT}
            ]
        }
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await request_with_retry(client, "POST", url, json=payload)
        response.raise_for_status()
        res_json = response.json()
        try:
            text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            usage_info = res_json.get("usageMetadata", {})
            usage = {
                "prompt_tokens": usage_info.get("promptTokenCount"),
                "completion_tokens": usage_info.get("candidatesTokenCount"),
                "total_tokens": usage_info.get("totalTokenCount")
            }
            # Gemini models by default return the thinking traces inside the text output if supported,
            # so we pass None for the reasoning field.
            return text, None, usage
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse Gemini response: {e}. Raw response: {res_json}")

async def call_openai_async(prompt: str, context: str, api_key: str, model: str, base_url: str | None, thinking: bool = True) -> tuple[str, str | None, dict[str, int | None]]:
    url = f"{base_url or 'https://api.openai.com/v1'}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    full_user_content = f"Context:\n{context}\n\nUser Question:\n{prompt}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_user_content}
        ],
        "temperature": 0.0
    }
    
    # Check if this is an NVIDIA NIM host
    if base_url and "integrate.api.nvidia.com" in base_url:
        # Enable the thinking toggle with default reasoning settings (enable_thinking is required by some NIM versions)
        payload["chat_template_kwargs"] = {
            "thinking": thinking,
            "enable_thinking": thinking
        }
        
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await request_with_retry(client, "POST", url, headers=headers, json=payload)
        response.raise_for_status()
        res_json = response.json()
        try:
            message = res_json["choices"][0]["message"]
            text = message.get("content", "")
            
            # Extract reasoning traces (commonly reasoning_content or reasoning)
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            
            usage_info = res_json.get("usage", {})
            usage = {
                "prompt_tokens": usage_info.get("prompt_tokens"),
                "completion_tokens": usage_info.get("completion_tokens"),
                "total_tokens": usage_info.get("total_tokens")
            }
            return text, reasoning, usage
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse OpenAI response: {e}. Raw response: {res_json}")

async def generate_llm_response(prompt: str, context: str, thinking: bool = True) -> tuple[str, str | None, dict[str, int | None]]:
    """
    Routes the LLM generation request to the configured LLM provider asynchronously,
    returning a tuple: (response_text, reasoning_text, token_usage_dict).
    """
    provider = settings.llm_provider
    model = settings.llm_model
    
    if provider == "gemini":
        api_key = settings.gemini_api_key or ""
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment settings.")
        return await call_gemini_async(prompt, context, api_key, model, thinking=thinking)
        
    elif provider == "openai":
        api_key = settings.openai_api_key or ""
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured in environment settings.")
        return await call_openai_async(prompt, context, api_key, model, settings.llm_base_url, thinking=thinking)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
