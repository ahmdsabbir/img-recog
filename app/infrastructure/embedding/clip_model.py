import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from app.interfaces.embedding import EmbeddingModel
from app.config import settings


class ClipEmbeddingModel(EmbeddingModel):
    def __init__(self, preprocessor=None):
        self.device = settings.DEVICE
        self.model = CLIPModel.from_pretrained(settings.EMBEDDING_MODEL).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(settings.EMBEDDING_MODEL)
        self.preprocessor = preprocessor

    def encode_image(
        self, image_path: str, save_preprocessed: bool = False, save_dir: str = "data/preprocessed"
    ) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")

        # Preprocess if preprocessor is available
        if self.preprocessor:
            image = self.preprocessor.preprocess(image)

            # Save preprocessed image if requested
            if save_preprocessed:
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.basename(image_path)
                preprocessed_path = os.path.join(save_dir, f"pre_{filename}")
                image.save(preprocessed_path)

        inputs = self.processor(images=image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        # Ensure tensor extraction
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        features = features.detach().cpu().numpy()[0]
        features = features / np.linalg.norm(features)

        return features.astype("float32")
