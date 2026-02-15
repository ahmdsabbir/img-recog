from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[int], vectors: np.ndarray):
        pass

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass