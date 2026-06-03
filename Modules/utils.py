

import numpy as np
from typing import Dict, List, Tuple, Set, Optional

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
