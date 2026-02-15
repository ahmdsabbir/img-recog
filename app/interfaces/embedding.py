from abc import ABC, abstractmethod
import numpy as np

class EmbeddingModel(ABC):

    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        pass
