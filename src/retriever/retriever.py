import faiss
import pickle
import numpy as np
from src.embedding.embeddings_model import EmbeddingModel
from src.utils.config import VECTOR_DB_PATH, TEXT_STORE_PATH, TOP_K

class Retriever:
    """
    Retrieves top-K relevant text chunks for a query.
    """

    def __init__(self):
        self.index = faiss.read_index(VECTOR_DB_PATH)
        with open(TEXT_STORE_PATH, "rb") as f:
            self.text_chunks = pickle.load(f)
        self.embedder = EmbeddingModel()

    def retrieve(self, query: str):
        query_vector = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vector), TOP_K)
        results = [self.text_chunks[i] for i in indices[0]]

        # Optional: filter chunks that contain at least one query keyword
        keywords = query.lower().split()
        filtered = []
        for r in results:
            if any(k in r.lower() for k in keywords):
                filtered.append(r)
        return filtered if filtered else results