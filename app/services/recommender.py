from app.interfaces.embedding import EmbeddingModel
from app.interfaces.vectore_store import VectorStore
from app.config import settings


class RecommenderService:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def recommend(self, image_path: str):
        vector = self.embedding_model.encode_image(image_path)
        ids, scores = self.vector_store.search(vector, settings.TOP_K)
        return ids, scores
