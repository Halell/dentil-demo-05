"""
Ollama client for LLM interactions.
Copied from main project for Block 1 usage.
"""
import os
import json
import httpx
from typing import Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
REASONING_LEVEL = os.getenv("OLLAMA_REASONING_LEVEL")  # high|medium|low

def _build_headers(api_key_env: str) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    api_key = os.getenv(api_key_env)
    if api_key:
        header_name = os.getenv("OLLAMA_AUTH_HEADER", "Authorization")
        if header_name.lower() == "authorization":
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers[header_name] = api_key
    return headers

def get_ollama_config(turbo: bool = False) -> Tuple[str, Dict[str, str], str]:
    base_var = "OLLAMA_TURBO_HOST" if turbo else "OLLAMA_HOST"
    base_url = (os.getenv(base_var) or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model_env = "OLLAMA_TURBO_MODEL" if turbo else "OLLAMA_MODEL"
    model = os.getenv("LLM_MODEL") or os.getenv(model_env) or DEFAULT_MODEL
    headers = _build_headers("OLLAMA_TURBO_KEY" if turbo else "OLLAMA_API_KEY") or _build_headers("OLLAMA_API_KEY")
    return base_url, headers, model

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat_json(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    json_only: bool = True,
    model_override: str = None,
    reasoning_level: str | None = None,
    turbo: bool = False,
) -> dict:
    """Call Ollama and get JSON response."""
    base_url, headers, model = get_ollama_config(turbo=turbo)
    if model_override:
        model = model_override
    
    if reasoning_level is None:
        reasoning_level = REASONING_LEVEL
    if reasoning_level:
        system = (system + f"\nReasoning: {reasoning_level}" ).strip()
    sys_msgs = ([{"role": "system", "content": system}] if system else [])
    user_msg = {"role": "user", "content": prompt}
    messages = sys_msgs + [user_msg]
    
    url = base_url + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature},
        "stream": False
    }
    
    if json_only:
        payload["format"] = "json"
    
    try:
        with httpx.Client(timeout=120, headers=headers) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            
            # Extract content
            content = data.get("message", {}).get("content", "{}")
            
            # Parse JSON
            return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": content}
    except Exception as e:
        return {"error": str(e)}