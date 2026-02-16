from app.interfaces.embedding import I_EmbeddingModel


class ZeroShotAttributeService:
    """Zero-shot attribute classification using CLIP text prompts"""

    # Zero-shot labels for different categories
    ATTRIBUTE_LABELS = {
        "shoe": {
            "type": ["a photo of a sneaker", "a photo of a boot", "a photo of a loafer", "a photo of a sandal"],
            "color": [
                "a photo of a black shoe",
                "a photo of a white shoe",
                "a photo of a red shoe",
                "a photo of a blue shoe",
                "a photo of a brown shoe",
                "a photo of a gray shoe",
                "a photo of a green shoe",
            ],
            "gender": ["a photo of a men's shoe", "a photo of a women's shoe", "a photo of a unisex shoe"],
            "age_group": ["a photo of an adult shoe", "a photo of a kid's shoe"],
        },
        "bag": {
            "type": ["a photo of a backpack", "a photo of a handbag", "a photo of a tote bag", "a photo of a clutch"],
            "color": [
                "a photo of a black bag",
                "a photo of a white bag",
                "a photo of a red bag",
                "a photo of a blue bag",
                "a photo of a brown bag",
            ],
            "style": ["a photo of a casual bag", "a photo of a formal bag"],
        },
    }

    def __init__(self, embedding_model: I_EmbeddingModel):
        self.embedding_model = embedding_model

    def classify(self, img_path: str, category: str):
        """Perform zero-shot classification for all attributes of a category"""
        attributes = self.ATTRIBUTE_LABELS.get(category)

        if not attributes:
            raise Exception(f"No zero-shot labels defined for category: {category}")

        results = {}

        for attr_name, labels in attributes.items():
            attr_results = self.embedding_model.classify_img_zeroshot(
                img_path=img_path, labels=labels
            )

            # Get best match
            best_label, confidence = attr_results[0]

            # Clean up label text (remove "a photo of a", "a photo of an", etc.)
            clean_label = best_label.replace("a photo of a ", "").replace("a photo of an ", "")

            results[attr_name] = {"value": clean_label, "confidence": confidence}

        return results
