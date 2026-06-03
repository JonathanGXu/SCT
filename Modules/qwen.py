

import os
import numpy as np
from typing import Any, Optional, Union, List, Dict

from openai import AsyncOpenAI

from lightrag.utils import wrap_embedding_func_with_attrs


DASHSCOPE_API_KEY = os.getenv("OPENAI_API_KEY")
DASHSCOPE_API_BASE = os.getenv(
    "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

qwen_async_client = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE,
)


async def _qwen_chat_completion(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    **kwargs,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await qwen_async_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    return response.choices[0].message.content.strip()


async def qwen_turbo_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    return await _qwen_chat_completion(
        "qwen-turbo", prompt, system_prompt, history_messages, **kwargs
    )


async def qwen_plus_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    return await _qwen_chat_completion(
        "qwen-plus", prompt, system_prompt, history_messages, **kwargs
    )


async def qwen_max_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    return await _qwen_chat_completion(
        "qwen-max", prompt, system_prompt, history_messages, **kwargs
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=2048, model_name="text-embedding-v2")
async def qwen_embed(
    texts: Union[str, List[str]],
    **kwargs,
) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]
    response = await qwen_async_client.embeddings.create(
        input=texts,
        model="text-embedding-v2",
    )
    embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
    return np.array(embeddings)

