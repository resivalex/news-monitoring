import numpy as np
from .types import MessageEmbedding


class ClusterProcessor:
    def __init__(self) -> None:
        self.stored_embeddings = []
        self.cluster_counter = 1
        self.cluster_ids = []

    def process_embedding(
        self, text_embedding: MessageEmbedding, similarity_threshold: float = 0.85
    ) -> (bool, int):
        """
        Check if the embedding is unique based on cosine similarity and assign a cluster ID.
        Returns a tuple of (is_unique, cluster_id).
        """
        for i, stored_vector in enumerate(self.stored_embeddings):
            cosine_similarity = self._calculate_similarity(
                text_embedding["embedding"], stored_vector["embedding"]
            )
            if cosine_similarity >= similarity_threshold:
                return False, self.cluster_ids[i]

        # If unique, assign a new cluster ID
        self.stored_embeddings.append(text_embedding)
        assigned_cluster = self.cluster_counter
        self.cluster_ids.append(assigned_cluster)
        self.cluster_counter += 1
        return True, assigned_cluster

    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        # Calculate cosine similarity between two vectors
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
