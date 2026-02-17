import os
import json
import torch

from app.interfaces.cache import I_Cache
from app.interfaces.embedding import I_EmbeddingModel
from app.models.attribute_head import AttributeHead
from app.config import settings
from app.infrastructure.cache.cache_keys import CacheKeys


class ProductAttributeService:
    def __init__(self, embedding_model: I_EmbeddingModel, cache: I_Cache):
        self.embedding_model = embedding_model
        self.device = settings.DEVICE
        self.cache = cache
    

    def _load_attribute_model(self, category: str, attribute: str):
        # See if it's cached or not
        chache_key = CacheKeys.attribute_model(category=category, attribute=attribute)

        cached = self.cache.get(chache_key)
        if cached:
            return cached
        
        # it's not chached
        base_path = f"models/{category}/{attribute}"

        model_path = os.path.join(base_path, "model.pt")
        classes_path = os.path.join(base_path, "classes.json")

        if not os.path.exists(model_path):
            return None

        with open(classes_path, "r") as f:
            classes = json.load(f)

        # Convert string keys to integers (JSON doesn't support integer keys)
        classes = {int(k): v for k, v in classes.items()}

        model = AttributeHead(
            embedding_dim=512,
            num_classes=len(classes)
        )

        model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        model.to(self.device)
        model.eval()

        # now cache it
        self.cache.set(chache_key, (model, classes))

        return model, classes


    def classify(self, img_path: str, category: str):
        embedding = self.embedding_model.encode_image(img_path)
        embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(self.device)

        category_dir = f"models/{category}"

        if not os.path.exists(category_dir):
            raise Exception(f"No trained models found for category: {category}")

        results = {}

        for attribute in os.listdir(category_dir):
            try:
                loaded = self._load_attribute_model(category, attribute)
            except Exception as e:
                print(f"Warning: Failed to load model for {category}/{attribute}: {e}")
                continue

            if not loaded:
                continue

            model, classes = loaded

            with torch.no_grad():
                logits = model(embedding_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()

            results[attribute] = {
                "value": classes[pred_idx],
                "confidence": float(probs[0][pred_idx].item())
            }

        return results
