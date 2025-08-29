import os, json, httpx
from typing import Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL", "os120b")

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

def get_ollama_config() -> Tuple[str, Dict[str, str], str]:
    base_url = (os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = DEFAULT_MODEL
    headers = _build_headers("OLLAMA_API_KEY") or _build_headers("OLLAMA_TURBO_KEY")
    return base_url, headers, model

def get_ollama_turbo_config() -> Tuple[str, Dict[str, str], str]:
    turbo_host = os.getenv("OLLAMA_TURBO_HOST")
    if not turbo_host:
        return get_ollama_config()
    turbo_url = turbo_host.rstrip("/")
    model = DEFAULT_MODEL
    headers = _build_headers("OLLAMA_TURBO_KEY") or _build_headers("OLLAMA_API_KEY")
    return turbo_url, headers, model

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat_json(
    prompt: str,
    system: str = "",
    temperature: float = 0.2,
    json_only: bool = True,
    turbo: bool = True,
    model_override: str | None = None,
    reasoning_level: str | None = None,
) -> dict:
    base_url, headers, model = get_ollama_turbo_config() if turbo else get_ollama_config()
    if model_override:
        model = model_override
    # Reasoning level injection: if provided (param beats env), prepend a line 'Reasoning: <level>' unless already present.
    reasoning_level = reasoning_level or os.getenv("OLLAMA_REASONING_DEFAULT")
    if reasoning_level:
        if not system:
            system = f"Reasoning: {reasoning_level}\nYou are a helpful assistant."
        elif "Reasoning:" not in system:
            system = f"Reasoning: {reasoning_level}\n" + system
    sys_msgs = ([{"role": "system", "content": system}] if system else [])
    user_msg = {"role": "user", "content": prompt}
    messages = sys_msgs + [user_msg]

    # Endpoint order can be overridden via env (comma separated base-relative paths)
    order_env = os.getenv("OLLAMA_ENDPOINT_ORDER")
    if order_env:
        endpoint_candidates = [p.strip() for p in order_env.split(',') if p.strip()]
    else:
        endpoint_candidates = [
            "/api/chat",              # modern Ollama chat
            "/api/generate",          # legacy generate
            "/v1/chat/completions",   # OpenAI-style
            "/v1/chat",               # alt chat
            "/v1/completions"         # OpenAI legacy
        ]

    last_error = {}
    data = {}
    with httpx.Client(timeout=120, headers=headers) as client:
        for ep in endpoint_candidates:
            url = base_url.rstrip('/') + ep
            try:
                if ep.endswith("/chat") or ep.endswith("/chat/completions") or ep.endswith("/api/chat"):
                    payload = {
                        "model": model,
                        "messages": messages,
                        "options": {"temperature": temperature},
                    }
                    if json_only:
                        payload["format"] = "json"
                elif ep.endswith("/generate"):
                    payload = {
                        "model": model,
                        "prompt": (system + "\n" if system else "") + prompt,
                        "stream": False,
                    }
                    if json_only:
                        payload["format"] = "json"
                else:  # generic OpenAI style
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "stream": False,
                    }
                r = client.post(url, json=payload)
                if r.status_code == 404:
                    # capture body for diagnostics (model missing vs path)
                    body_snip = r.text[:200]
                    last_error[ep] = {"status": 404, "body": body_snip}
                    # If body hints model not found – stop early with clear error
                    if 'not found' in body_snip.lower() and model in body_snip:
                        raise RuntimeError(f"Model '{model}' not found on server. Install/pull it first (e.g. ollama pull {model}). Server reply: {body_snip}")
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:  # noqa
                last_error[ep] = str(e)
        else:
            raise RuntimeError(f"All candidate endpoints failed: {last_error}")

    # Normalize content across variants
    content = None
    if isinstance(data, dict):
        content = (
            data.get("message", {}).get("content") or
            data.get("response") or
            data.get("choices", [{}])[0].get("message", {}).get("content")
        )
    if content is None:
        content = "{}"
    # Hard JSON parse; if fails, wrap in {} with raw text
    try:
        return json.loads(content)
    except Exception:
        return {"raw": content, "diagnostic": data}

def detect_server() -> dict:
    base_url, headers, _ = get_ollama_config()
    info = {"base_url": base_url}
    try:
        r = httpx.get(base_url, timeout=5, headers=headers)
        info["root_status"] = r.status_code
        info["root_snippet"] = r.text[:120]
    except Exception as e:  # noqa
        info["error_root"] = str(e)
    # collect body for tags & version
    paths = ["/api/tags", "/api/version", "/api/models", "/api/show"]
    for path in paths:
        try:
            r = httpx.get(base_url + path, timeout=5, headers=headers)
            entry = {"status": r.status_code}
            if path == "/api/tags" and r.status_code == 200:
                try:
                    tags_json = r.json()
                    entry["models"] = [t.get("name") for t in tags_json.get("models", [])]
                except Exception:
                    entry["body"] = r.text[:500]
            elif path == "/api/version" and r.status_code == 200:
                entry["body"] = r.text[:200]
            info[path] = entry
        except Exception as e:  # noqa
            info[path] = f"ERR:{e}" 
    return info

def list_models() -> list[str]:
    base_url, headers, _ = get_ollama_config()
    try:
        r = httpx.get(base_url + "/api/tags", timeout=10, headers=headers)
        if r.status_code == 200:
            js = r.json()
            return [m.get("name") for m in js.get("models", [])]
    except Exception:
        return []
    return []

def generate_simple(prompt: str, model: str | None = None, turbo: bool = True, temperature: float = 0.7, max_tokens: int = 256) -> str:
    base_url, headers, default_model = get_ollama_turbo_config() if turbo else get_ollama_config()
    model = model or default_model
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    r = httpx.post(base_url.rstrip('/') + "/api/generate", json=payload, headers=headers, timeout=120)
    if r.status_code == 404:
        body = r.text[:300]
        raise RuntimeError(f"/api/generate 404. Model installed? ({model}) Body: {body}")
    r.raise_for_status()
    js = r.json()
    return js.get("response", "")

__all__ = [
    "chat_json",
    "generate_simple",
    "list_models",
    "get_ollama_turbo_config",
    "get_ollama_config",
    "detect_server"
]

__all__ = ["chat_json", "get_ollama_turbo_config", "get_ollama_config"]
