
import os
from http import HTTPStatus
from openai import OpenAI

from Modules.ollama_cloud import ollama_chat_completion
from config import DeepSeek_key

QWEN_API_KEY = "sk-xxx"



def qwen_chat_completion(question, model='qwen-flash', use_ollama=False, ollama_chat_model: str = "glm-5:cloud", use_deepseek=False, prompt=None):

    if prompt is not None:
        final_prompt = prompt
    else:
        final_prompt = f"""
            Answer the following question：{question}
            """
    try:
        if use_ollama:
            answer = ollama_chat_completion(final_prompt, model=ollama_chat_model)
            print(answer)
            return answer

        if use_deepseek:
            client = OpenAI(
                api_key=DeepSeek_key,
                base_url="https://api.deepseek.com/v1"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1,
                timeout=600
            )
            answer = response.choices[0].message.content.strip()
            print(answer)
            return answer

        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.1,
            timeout=600
        )
        answer = response.choices[0].message.content.strip()
        print(answer)
        return answer

    except Exception as e:
        raise Exception(f"Fail: {str(e)}")
