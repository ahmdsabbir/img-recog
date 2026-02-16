import os
import faiss
import numpy as np

from app.interfaces.vectore_store import I_VectorStore
from app.config import settings


class FaissVectorStore(I_VectorStore):
    def __init__(self):
        self.index_path = settings.FAISS_INDEX_PATH
        self.dimension = settings.EMBEDDING_DIM
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []

    def add(self, ids, vectors):
        self.index.add(vectors)
        self.id_map.extend(ids)

    def search(self, vector, top_k):
        vector = np.expand_dims(vector, axis=0)
        distances, indices = self.index.search(vector, top_k)
        result_ids = [self.id_map[i] for i in indices[0]]
        scores = distances[0].tolist()
        return result_ids, scores

    def save(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.index_path + "_ids.npy", np.array(self.id_map))

    def load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            self.id_map = np.load(self.index_path + "_ids.npy").tolist()
