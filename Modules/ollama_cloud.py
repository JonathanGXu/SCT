

import os
import time
import numpy as np
from ollama import Client
from config import Ollama_API_key

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", Ollama_API_key)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "glm-5:cloud")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

_ollama_client = Client(
    host=OLLAMA_HOST,
    headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY},
    timeout=300,
)


def _retry_call(func, max_retries=3, base_delay=2.0):

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not support" in error_msg or "no such file" in error_msg:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            print(f"⚠️ Ollama Fail (attempt {attempt}/{max_retries}): {e}，{wait}s ")
            time.sleep(wait)
    raise Exception(f"Ollama  {max_retries} Fail: {last_exception}")


def ollama_chat_completion(prompt: str, model: str = None, stream: bool = False) -> str:

    model = model or DEFAULT_CHAT_MODEL
    messages = [{'role': 'user', 'content': prompt}]

    def _call():
        if stream:
            response_text = ""
            for part in _ollama_client.chat(model, messages=messages, stream=True):
                response_text += part['message']['content']
            return response_text
        else:
            response = _ollama_client.chat(model, messages=messages, stream=False)
            if isinstance(response, dict):
                return response['message']['content']
            else:
                return response.message.content

    try:
        return _retry_call(_call, max_retries=3, base_delay=2.0)
    except Exception as e:
        raise Exception(f" Ollama Cloud Chat API Fail (model={model}): {str(e)}")


def ollama_embed(texts: list[str], model: str = None) -> np.ndarray:

    model = model or DEFAULT_EMBED_MODEL
    embeddings = []

    for text in texts:
        def _call():
            response = _ollama_client.embed(model=model, input=text)
            if isinstance(response, dict):
                return np.array(response["embedding"], dtype=np.float32)
            else:
                return np.array(response.embedding, dtype=np.float32)

        try:
            emb = _retry_call(_call, max_retries=3, base_delay=2.0)
            embeddings.append(emb)
        except Exception as e:
            raise Exception(f"Ollama Cloud Embed API Fail (model={model}, text={text[:30]}...): {str(e)}")

    return np.array(embeddings)
