from app.interfaces.embedding import I_EmbeddingModel


class CategoryClassifierService:
    def __init__(self, embedding_model: I_EmbeddingModel):
        self.embedding_model = embedding_model

        self.labels = [
            "a photo of a shoe",
            "a photo of a bag",
        ]

    
    def classify(self, img_path: str):
        results = self.embedding_model.classify_img(img_path=img_path, labels=self.labels)
        best_label, confidence = results[0]

        category = best_label.replace("a photo of a ", "")

        return category, confidence