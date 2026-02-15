from app.interfaces.embedding import EmbeddingModel
from app.interfaces.vectore_store import VectorStore
from app.config import settings


class RecommenderService:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def recommend(self, image_path: str, save_preprocessed: bool = False, save_dir: str = "data/preprocessed"):
        vector = self.embedding_model.encode_image(
            image_path, save_preprocessed=save_preprocessed, save_dir=save_dir
        )
        ids, scores = self.vector_store.search(vector, settings.TOP_K)
        return ids, scores
