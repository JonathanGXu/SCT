import os
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from typing import List, Dict, Tuple, Set

from .Agent import GraphRAGAgent

Triple = Tuple[str, str, str]
AgentID = str
ConsensusScore = float


class EMGraphConsensus:

    def __init__(self, agent_num: int = 3, agent_model:str = "qwen-turbo", use_ollama: bool = False, ollama_chat_model: str = "glm-5:cloud", use_deepseek: bool = False):

        self.agents = [
            GraphRAGAgent(f"agent_{i}", agent_model=agent_model, use_ollama=use_ollama, ollama_chat_model=ollama_chat_model, use_deepseek=use_deepseek)
            for i in range(agent_num)
        ]
        self.all_triples: Dict[Triple, Set[AgentID]] = {}

        self.convergence_threshold = 1e-3

    def _similarity(self, triple: Triple, text: str) -> float:

        h, r, b = triple
        score = 0.0
        if h in text: score += 0.3
        if r in text: score += 0.4
        if b in text: score += 0.3
        return max(score, 0.3)

    def e_step(self, text: str) -> Dict[Triple, ConsensusScore]:
        triple_consensus = {}

        for triple, agent_ids in self.all_triples.items():
            sim = self._similarity(triple, text)
            z_sum = 0.0

            for agent_id in agent_ids:
                agent = next(a for a in self.agents if a.agent_id == agent_id)
                z_i_t = agent.credibility * sim
                z_sum += z_i_t

            triple_consensus[triple] = z_sum / len(agent_ids)
        return triple_consensus

    def m_step(self, text: str, triple_consensus: Dict[Triple, ConsensusScore]):

        for agent in self.agents:

            agent_triples = [t for t, aids in self.all_triples.items() if agent.agent_id in aids]
            if not agent_triples:
                continue

            numerator = 0.0
            denominator = 0.0
            for triple in agent_triples:
                sim = self._similarity(triple, text)
                numerator += triple_consensus[triple] * sim
                denominator += sim


            alpha_prior = 1.0
            prior_mean = 0.5
            numerator += alpha_prior * prior_mean
            denominator += alpha_prior


            new_credibility = numerator / denominator if denominator != 0 else 0.5

            new_credibility = max(0.1, min(1.0, new_credibility))
            agent.credibility = new_credibility

    def run_em_iteration(self, text: str) -> bool:

        old_credibilities = [a.credibility for a in self.agents]

        triple_consensus = self.e_step(text)

        self.m_step(text, triple_consensus)

        new_credibilities = [a.credibility for a in self.agents]
        max_diff = max(abs(new - old) for new, old in zip(new_credibilities, old_credibilities))
        return max_diff < self.convergence_threshold

    def train(self, text: str, max_iter: int = 100) -> Tuple[Dict[Triple, ConsensusScore], List[GraphRAGAgent]]:

        for agent in self.agents:
            triples = agent.extract_triples(text)
            for triple in triples:
                if triple not in self.all_triples:
                    self.all_triples[triple] = set()
                self.all_triples[triple].add(agent.agent_id)

        for iter_num in range(max_iter):
            is_converged = self.run_em_iteration(text)
            if is_converged:
                break

        final_consensus = self.e_step(text)
        return final_consensus, self.agents

def build_consensus_subgraph(final_consensus: Dict[Triple, ConsensusScore]) -> nx.DiGraph:

    scores = list(final_consensus.values())
    median_score = np.median(scores)

    consensus_triples = {t: s for t, s in final_consensus.items() if s >= median_score}

    g = nx.DiGraph()
    for (h, r, b), score in consensus_triples.items():
        g.add_edge(h, b, relation=r, consensus_score=score)
    return g


def build_consensus_tensor(final_consensus: Dict[Triple, ConsensusScore]) -> coo_matrix:

    entities = set()
    relations = set()
    for h, r, b in final_consensus.keys():
        entities.add(h)
        entities.add(b)
        relations.add(r)

    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    h_ids, r_ids, b_ids = [], [], []
    values = []
    for (h, r, b), score in final_consensus.items():
        h_ids.append(ent2id[h])
        r_ids.append(rel2id[r])
        b_ids.append(ent2id[b])
        values.append(score)

    shape = (len(ent2id), len(rel2id) * len(ent2id))
    flat_rb_ids = [rel2id[r] * len(ent2id) + ent2id[b] for h, r, b in final_consensus.keys()]
    tensor = coo_matrix((values, (h_ids, flat_rb_ids)), shape=shape)

    return tensor, ent2id, rel2id


