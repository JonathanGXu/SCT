

import networkx as nx

from Modules.Graphs import EMGraphConsensus, build_consensus_subgraph, build_consensus_tensor
from Modules.Storage import GraphRAGStorage, GraphRAGVectorIndex, GraphRAGRetriever
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from Modules.utils import normalize_embedding
from openai import OpenAI

QWEN_API_KEY = "sk-XXX"
class Graphrag_em:
    def __init__(
            self,
            agent_num:int=3,
            agent_model:str="qwen-turbo",
            embedding_model:str="text-embedding-v4",
            qa_model="qwen-turbo",
            similarity_threshold = 0.8,
            use_ollama: bool = False,
            ollama_chat_model: str = "glm-5:cloud",
            ollama_embed_model: str = "mxbai-embed-large",
            use_deepseek: bool = False
            ):
        self.agent_num = agent_num
        self.agent_model = agent_model
        self.retriever_emb_model = embedding_model
        self.qa_model = qa_model
        self.sim_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.use_ollama = use_ollama
        self.ollama_chat_model = ollama_chat_model
        self.ollama_embed_model = ollama_embed_model
        self.use_deepseek = use_deepseek

        self.client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


    def build_graph(self,input_text):

        em_trainer = EMGraphConsensus(
            agent_num=self.agent_num, agent_model=self.agent_model,
            use_ollama=self.use_ollama, ollama_chat_model=self.ollama_chat_model,
            use_deepseek=self.use_deepseek
        )

        all_consensus = []
        for i, text in enumerate(input_text):

            final_consensus, agents = em_trainer.train(text)

            all_consensus.append(final_consensus)

        disambiguated_result = self._merge_synonym_triples(all_consensus)


        consensus_graph = build_consensus_subgraph(disambiguated_result)

        self.consensus_graph = consensus_graph

        consensus_tensor, ent2id, rel2id = build_consensus_tensor(disambiguated_result)
        print("\n======")
        print(f"entity ID: {ent2id}")
        print(f"relation ID: {rel2id}")
        print(f"Tensor shape: {consensus_tensor.shape}")


        storage = GraphRAGStorage()
        storage.clear_old_graph()
        storage.store_consensus_subgraph(self.consensus_graph)

        vector_index = GraphRAGVectorIndex(embedding_model=self.embedding_model, ollama_embed_model=self.ollama_embed_model)
        vector_index.build_index(self.consensus_graph)

        self.retriever = GraphRAGRetriever(
            qa_model=self.qa_model, use_ollama=self.use_ollama,
            ollama_chat_model=self.ollama_chat_model, ollama_embed_model=self.ollama_embed_model,
            use_deepseek=self.use_deepseek
        )

    def qa(self, question):

        answer,evidence = self.retriever.qa_with_graphrag(question)

        return answer, evidence

    def _merge_synonym_triples(self,all_consensus):

        triple_score_pool = {}
        for consensus in all_consensus:
            for triple, score in consensus.items():
                if triple not in triple_score_pool:
                    triple_score_pool[triple] = []
                triple_score_pool[triple].append(score)

        final_consensus =  {}
        for triple, scores in triple_score_pool.items():

            avg_score = np.mean(scores)
            final_score = float(avg_score)

            final_consensus[triple] = final_score

        return final_consensus


def normalize_entity(entity: str) -> str:
    if not entity:
        return ""

    normalized = entity.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


    return normalized

def normalize_relation(relation: str) -> str:
    if not relation:
        return ""
    normalized = relation.strip().lower().replace(" ", "").replace("-", "")

    return normalized