from app.config import settings
from app.interfaces.embedding import I_EmbeddingModel
from app.interfaces.vectore_store import I_VectorStore


class RecommenderService:
    def __init__(self, embedding_model: I_EmbeddingModel, vector_store: I_VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def recommend(self, image_path: str, save_preprocessed: bool = False, save_dir: str = "data/preprocessed"):
        vector = self.embedding_model.encode_image(
            image_path, save_preprocessed=save_preprocessed, save_dir=save_dir
        )
        ids, scores = self.vector_store.search(vector, settings.TOP_K)

        return ids, scores
