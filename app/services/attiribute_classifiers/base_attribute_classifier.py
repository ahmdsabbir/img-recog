from abc import ABC, abstractmethod
from app.interfaces.embedding import I_EmbeddingModel

class BaseAttributeClassifier(ABC):
    def __init__(self, embedding_model: I_EmbeddingModel):
        self.embedding_model = embedding_model

    @abstractmethod
    def get_schema(self) -> dict:
        """
        Returns attribute schema.
        example:
        {
            "color": [...]
            "gender": [...]
        }
        """
        pass


    def classify(self, img_path: str):
        results = {}
        schema = self.get_schema()

        for attr_name, labels in schema.items():
            classification = self.embedding_model.classify_img(img_path=img_path, labels=labels)
            best_label, confidence = classification[0]

            results[attr_name] = {
                "label": best_label,
                "confidence": float(confidence)
            }

        return results