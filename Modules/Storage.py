
import os
import numpy as np
import faiss
import json
import re
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv
from py2neo import Graph, Node, Relationship
from Modules.ollama_cloud import ollama_chat_completion, ollama_embed
from config import DeepSeek_key

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "88888888"

QWEN_API_KEY = "sk-b6599be971d84854b59c10a1f7b625a8"

VECTOR_INDEX_PATH = "../Storage/graphrag_vector_index"
ENTITY_EMBEDDING_PATH = "../Storage/entity_embeddings.json"

EMBEDDING_DIM = 1024


def normalize_embedding(embedding: List[float], target_dim: int = EMBEDDING_DIM) -> np.ndarray:
    emb_np = np.array(embedding, dtype=np.float32)
    if len(emb_np) != target_dim:
        if len(emb_np) > target_dim:
            emb_np = emb_np[:target_dim]
        else:
            emb_np = np.pad(emb_np, (0, target_dim - len(emb_np)), mode='constant')
    norm = np.linalg.norm(emb_np)
    if norm == 0:
        emb_np = np.ones(target_dim, dtype=np.float32) / np.sqrt(target_dim)
    else:
        emb_np = emb_np / norm
    return emb_np


def clean_entity_name(name: str) -> str:
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', name).strip()


class GraphRAGStorage:

    def __init__(self):
        self.graph_db = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def clear_old_graph(self):

        self.graph_db.run("MATCH (n) DETACH DELETE n")


    def store_consensus_subgraph(self, consensus_graph: nx.DiGraph):

        entities = set()
        for u, v in consensus_graph.edges():
            entities.add(u)
            entities.add(v)

        for entity in entities:
            self.graph_db.run("MERGE (e:Entity {name: $name})", name=entity)
        print(f"✅ Storage Entity:{list(entities)}")

        rel_count = 0
        for u, v, data in consensus_graph.edges(data=True):
            rel_type = data["relation"]
            consensus_score = data["consensus_score"]

            cypher = f"""
            MATCH (e1:Entity {{name: $head}}), (e2:Entity {{name: $tail}})
            MERGE (e1)-[r:`{rel_type}`]->(e2)
            SET r.consensus_score = $score
            """
            self.graph_db.run(
                cypher,
                head=u,
                tail=v,
                score=consensus_score
            )
            rel_count += 1


    def load_consensus_subgraph(self) -> nx.DiGraph:

        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN e1.name as head, type(r) as relation, e2.name as tail, r.consensus_score as score
        """
        results = self.graph_db.run(query).data()

        g = nx.DiGraph()
        for res in results:
            g.add_edge(
                res["head"],
                res["tail"],
                relation=res["relation"],
                consensus_score=res["score"]
            )
        return g


class GraphRAGVectorIndex:

    def __init__(self, embedding_model="text-embedding-v4", ollama_embed_model: str = "mxbai-embed-large"):
        self.embedding_model = embedding_model
        self.ollama_embed_model = ollama_embed_model
        self.use_ollama_embed = (embedding_model.lower() in ["ollama", "ollama_embed", "ollama-embed"])

        self.client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.index: Optional[faiss.IndexFlatL2] = None
        self.id2entity: List[str] = []

    def get_entity_embedding(self, text: str) -> np.ndarray:

        try:
            if self.use_ollama_embed:
                emb = ollama_embed([text], model=self.ollama_embed_model)[0]
                embedding = normalize_embedding(emb.tolist(), EMBEDDING_DIM)
            else:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                    encoding_format="float"
                )
                embedding_list = response.data[0].embedding
                embedding = normalize_embedding(embedding_list, EMBEDDING_DIM)
            return embedding
        except Exception as e:
            random_emb = np.random.rand(EMBEDDING_DIM).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            return random_emb

    def build_index(self, consensus_graph: nx.DiGraph):

        entities = set()
        for u, v in consensus_graph.edges():
            entities.add(u)
            entities.add(v)
        entities = list(entities)

        embeddings = []
        for entity in entities:
            embedding = self.get_entity_embedding(entity)
            self.entity_embeddings[entity] = embedding
            embeddings.append(embedding)
            self.id2entity.append(entity)

        embeddings_np = np.array(embeddings, dtype=np.float32)
        assert embeddings_np.shape == (len(entities), EMBEDDING_DIM)

        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.index.add(embeddings_np)


        os.makedirs(VECTOR_INDEX_PATH, exist_ok=True)
        faiss.write_index(self.index, os.path.join(VECTOR_INDEX_PATH, "entity_index.faiss"))
        with open(ENTITY_EMBEDDING_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "entity_embeddings": {k: v.tolist() for k, v in self.entity_embeddings.items()},
                "id2entity": self.id2entity,
                "embedding_dim": EMBEDDING_DIM,
                "embedding_model": "text-embedding-v4"
            }, f, ensure_ascii=False)


    def load_index(self):

        self.index = faiss.read_index(os.path.join(VECTOR_INDEX_PATH, "entity_index.faiss"))
        with open(ENTITY_EMBEDDING_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entity_embeddings = {
            k: normalize_embedding(v, data["embedding_dim"])
            for k, v in data["entity_embeddings"].items()
        }
        self.id2entity = data["id2entity"]

    def retrieve_similar_entities(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:

        query_embedding = self.get_entity_embedding(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self.id2entity):
                entity = self.id2entity[idx]
                similarity = 1 / (1 + dist)
                results.append((entity, round(similarity, 4)))

        return results


class GraphRAGRetriever:

    def __init__(self, qa_model="qwen-turbo", use_ollama: bool = False, ollama_chat_model: str = "glm-5:cloud", ollama_embed_model: str = "mxbai-embed-large", use_deepseek: bool = False):
        self.qa_model = qa_model
        self.use_ollama = use_ollama
        self.use_deepseek = use_deepseek
        self.ollama_chat_model = ollama_chat_model
        self.storage = GraphRAGStorage()
        self.vector_index = GraphRAGVectorIndex(ollama_embed_model=ollama_embed_model)
        self.consensus_graph: Optional[nx.DiGraph] = None


        self.consensus_graph = self.storage.load_consensus_subgraph()
        self.vector_index.load_index()

    def extract_subgraph(self, core_entities: List[str], hop: int = 2) -> nx.DiGraph:

        print(f"🔍 Entity：{core_entities}")
        print(f"🔍 Entity in subgraph:{list(self.consensus_graph.nodes)}")


        clean2original = {clean_entity_name(n): n for n in self.consensus_graph.nodes}
        cleaned_query_entities = [clean_entity_name(e) for e in core_entities]

        valid_entities = []
        for clean_e, original_e in zip(cleaned_query_entities, core_entities):

            if clean_e in clean2original:
                valid_entities.append(clean2original[clean_e])
            else:

                for n in self.consensus_graph.nodes:
                    if clean_e in clean_entity_name(n):
                        valid_entities.append(n)
                        break

        print(f"🔍 Entity：{valid_entities}")

        if not valid_entities:
            print("⚠️ Return empty")
            return nx.DiGraph()


        core_nodes = set(valid_entities)
        for _ in range(hop):
            new_nodes = set()
            for node in core_nodes:
                if node in self.consensus_graph:
                    new_nodes.update(self.consensus_graph.successors(node))
                    new_nodes.update(self.consensus_graph.predecessors(node))
            core_nodes.update(new_nodes)

        subgraph = self.consensus_graph.subgraph(core_nodes).copy()
        print(f"📊 Subgraph:{len(subgraph.nodes)} Entities，{len(subgraph.edges)} Relation")
        return subgraph

    def subgraph_to_prompt(self, subgraph: nx.DiGraph) -> str:

        if len(subgraph.edges()) == 0:
            return "[Knowledge Graph Information]\nNo relevant knowledge graph information is available to answer this question."
        prompt_lines = ["[Knowledge Graph Information]"]
        prompt_lines.append(f"The following is a relevant subgraph retrieved from the knowledge graph, containing {len(subgraph.nodes)} entities and {len(subgraph.edges)} relations:")
        prompt_lines.append("")
        for i, (u, v, data) in enumerate(subgraph.edges(data=True), 1):
            prompt_lines.append(f"{i}. {u} → [{data['relation']}] → {v}")
        prompt_lines.append("")
        prompt_lines.append("Hint: You may use the relationship chains among the above entities to perform multi-hop reasoning to answer the question.")
        return "\n".join(prompt_lines)

    def record_evidence(self, subgraph: nx.DiGraph):
        if len(subgraph.edges()) == 0:
            return []

        evidences = []
        for u, v, data in subgraph.edges(data=True):

            evidences.append([u,data['relation'],v])

        return evidences

    def qa_with_graphrag(self, query: str, top_k: int = 3, hop: int = 2) -> str:

        similar_entities = self.vector_index.retrieve_similar_entities(query, top_k=top_k)
        print(f"\n🔍 Entity：{[e[0] for e in similar_entities]}")

        core_entities = [e[0] for e in similar_entities]
        subgraph = self.extract_subgraph(core_entities, hop=hop)

        evidence = self.record_evidence(subgraph)

        subgraph_prompt = self.subgraph_to_prompt(subgraph)
        print(f"\n ❓️ Question {query}")
        print(f"\n 📊 Subgraph {subgraph_prompt}")
        final_prompt = f"""{subgraph_prompt}

[Question]
{query}

[Answer Requirements]
1. Answer the question STRICTLY based on the information in the knowledge graph above. Do NOT introduce external knowledge.
2. Use the relationship chains among entities for reasoning (e.g., A → [relation] → B → [relation] → C multi-hop relations).
3. If the graph information is sufficient to answer the question, provide a concise and accurate answer.
4. If the graph information is INSUFFICIENT to answer the question, explicitly state: "Based on the provided knowledge graph information, this question cannot be answered." Do NOT make up answers.
5. Keep the answer as brief as possible without redundant explanatory text.
"""

        if self.use_ollama:
            answer = ollama_chat_completion(final_prompt, model=self.ollama_chat_model)
        elif self.use_deepseek:
            client = OpenAI(
                api_key=DeepSeek_key,
                base_url="https://api.deepseek.com/v1"
            )
            response = client.chat.completions.create(
                model=self.qa_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1,
                timeout=600
            )
            answer = response.choices[0].message.content.strip()
        else:
            client = OpenAI(
                api_key=QWEN_API_KEY,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            response = client.chat.completions.create(
                model=self.qa_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1,
                timeout=600
            )
            answer = response.choices[0].message.content.strip()
        return answer, evidence

