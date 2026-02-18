from app.config import settings
from app.services.recommender import RecommenderService
from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.infrastructure.vector_store.faiss_store import FaissVectorStore
from app.infrastructure.preprocessing.factory import make_preprocessor
from app.infrastructure.cache.chache import Cache


class Container:
    """
    Constructs and holds shared infrastructure objects.
    Swap out any component here without touching command handlers
    """

    def __init__(self):
        preprocessor = make_preprocessor(settings)
        self.embedding = ClipEmbeddingModel(preprocessor=preprocessor)
        self.vectore_store = FaissVectorStore()
        self.recommender = RecommenderService(self.embedding, self.vectore_store)
        self.cache = Cache()
