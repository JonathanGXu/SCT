
from openai import OpenAI
from ollama import Client
from typing import List, Tuple

from Modules.ollama_cloud import ollama_chat_completion
from config import DeepSeek_key

Triple = Tuple[str, str, str]
AgentID = str
ConsensusScore = float
OLLAMA_API_KEY = "XXX"

class GraphRAGAgent:
    def __init__(self, agent_id: AgentID, agent_model= "qwen-turbo", use_ollama=False, ollama_chat_model: str = "glm-5:cloud", use_deepseek=False):
        self.agent_model = agent_model
        self.agent_id = agent_id
        self.credibility = 0.5  # 初始可信度 θ_i^(0) = 0.5
        self.use_ollama = use_ollama
        self.ollama_chat_model = ollama_chat_model
        self.use_deepseek = use_deepseek

    def extract_triples(self, text: str) -> List[Triple]:

        prompt = f"""
        You are a professional knowledge graph triple extraction assistant.
        Strictly extract all core triples from the given text in the format: (head entity, relation, tail entity).
        Output ONLY the result triples without any extra explanation, notes, or extra text.

        Text: {text}
        
        Example output: (Artificial Intelligence, is a, technical science)
        (Machine Learning, is a branch of, Artificial Intelligence)
        """

        if self.use_ollama:
            try:
                triples_str = ollama_chat_completion(prompt, model=self.ollama_chat_model)
            except Exception as e:
                print(f"Agent {self.agent_id} Ollama Fail: {e}")
                return []
        elif self.use_deepseek:
            client = OpenAI(
                api_key=DeepSeek_key,
                base_url="https://api.deepseek.com/v1",
            )
            try:
                response = client.chat.completions.create(
                    model=self.agent_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=600
                )
                triples_str = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Agent {self.agent_id} Deepseek 抽取失败: {e}")
                return []
        else:
            client = OpenAI(
                api_key="XXX",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            try:
                response = client.chat.completions.create(
                    model=self.agent_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    timeout=600
                )
                triples_str = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Agent {self.agent_id} Fail: {e}")
                return []

        triples = []
        for line in triples_str.split("\n"):
            line = line.strip().strip("()").replace("，", ",").replace(" ", "")
            parts = line.split(",")
            if len(parts) == 3:
                h, r, b = parts
                triples.append((h.strip(), r.strip(), b.strip()))
        return triples
