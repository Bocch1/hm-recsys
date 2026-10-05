"""
Embedding-based candidate retrieval module supporting visual, textual, and fused representations.
"""

from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class EmbeddingRecall:
    """Vector similarity retrieval engine combining visual and textual representations."""

    def __init__(self, image_emb, text_emb, user_purchased_hist):
        """
        Args:
            image_emb (dict): Mapping {article_id: np.array(dim)}.
            text_emb (dict): Mapping {article_id: np.array(dim)}.
            user_purchased_hist (dict): Mapping {uid: set(article_id)}.
        """
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.user_purchased_hist = user_purchased_hist

        self._build_matrices()

    def _build_matrices(self):
        """Constructs aligned continuous embedding matrices for vectorized nearest-neighbor search."""
        if self.image_emb:
            self.image_ids = list(self.image_emb.keys())
            self.image_matrix = np.vstack([self.image_emb[a] for a in self.image_ids])
        else:
            self.image_ids = []
            self.image_matrix = None

        if self.text_emb:
            self.text_ids = list(self.text_emb.keys())
            self.text_matrix = np.vstack([self.text_emb[a] for a in self.text_ids])
        else:
            self.text_ids = []
            self.text_matrix = None

        print(f"Embedding matrices indexed: image={self.image_matrix.shape if self.image_matrix is not None else 'None'}, "
              f"text={self.text_matrix.shape if self.text_matrix is not None else 'None'}")

    def _user_embedding(self, uid, purchased, emb_dict, emb_matrix, emb_ids, last_n=10):
        """Computes customer profile vector by averaging embeddings of recent purchases."""
        if not purchased:
            return None
        purchased_list = list(purchased)[-last_n:]
        valid = [emb_dict[a] for a in purchased_list if a in emb_dict]
        if not valid:
            return None
        user_emb = np.mean(valid, axis=0)
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm
        return user_emb

    def recall_image(self, uid, purchased, top_k=200):
        """Retrieves candidates based on visual similarity."""
        if self.image_matrix is None:
            return []
        user_emb = self._user_embedding(uid, purchased, self.image_emb,
                                        self.image_matrix, self.image_ids)
        if user_emb is None:
            return []

        scores = self.image_matrix.dot(user_emb)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            aid = self.image_ids[idx]
            if aid not in purchased:
                results.append(aid)
                if len(results) >= top_k:
                    break
        return results

    def recall_text(self, uid, purchased, top_k=200):
        """Retrieves candidates based on text semantic similarity."""
        if self.text_matrix is None:
            return []
        user_emb = self._user_embedding(uid, purchased, self.text_emb,
                                        self.text_matrix, self.text_ids)
        if user_emb is None:
            return []

        scores = self.text_matrix.dot(user_emb)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            aid = self.text_ids[idx]
            if aid not in purchased:
                results.append(aid)
                if len(results) >= top_k:
                    break
        return results

    def recall_fusion(self, uid, purchased, top_k=300):
        """Retrieves candidates by fusing normalized visual and text similarity scores."""
        if self.image_matrix is None and self.text_matrix is None:
            return []

        user_image_emb = self._user_embedding(uid, purchased, self.image_emb,
                                              self.image_matrix, self.image_ids)
        user_text_emb = self._user_embedding(uid, purchased, self.text_emb,
                                             self.text_matrix, self.text_ids)

        scores = defaultdict(float)

        if user_image_emb is not None:
            image_scores = self.image_matrix.dot(user_image_emb)
            for i, aid in enumerate(self.image_ids):
                scores[aid] += 0.5 * image_scores[i]

        if user_text_emb is not None:
            text_scores = self.text_matrix.dot(user_text_emb)
            for i, aid in enumerate(self.text_ids):
                scores[aid] += 0.5 * text_scores[i]

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for aid, _ in sorted_items:
            if aid not in purchased:
                results.append(aid)
                if len(results) >= top_k:
                    break
        return results

    def recall_batch(self, user_ids, purchased_hist, top_k=300, batch_size=2000):
        """Performs batch candidate retrieval across target customer identifiers."""
        results = {}
        users = list(user_ids)
        for i in tqdm(range(0, len(users), batch_size), desc="Embedding retrieval", leave=False):
            batch = users[i:i + batch_size]
            for uid in batch:
                purchased = purchased_hist.get(uid, set())
                results[uid] = np.array(
                    self.recall_fusion(uid, purchased, top_k=top_k),
                    dtype=np.int32
                )
        return results