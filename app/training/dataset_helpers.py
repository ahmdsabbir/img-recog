# app/training/dataset_helpers.py
import json
import torch
from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
import os


def load_embeddings_and_labels(dataset_json, products_dir="data/products"):
    """
    Generic function to compute CLIP embeddings and return labels.
    Works for any product type (shoes, bags, etc.)
    """
    clip_model = ClipEmbeddingModel()
    embeddings = []
    labels_dict = {}

    with open(dataset_json) as f:
        data = json.load(f)

    # Determine label keys dynamically
    label_keys = [k for k in data[0].keys() if k != "filename"]
    labels_dict = {k: [] for k in label_keys}

    for sample in data:
        path = os.path.join(products_dir, sample["filename"])
        emb = clip_model.encode_image(path)
        embeddings.append(emb)

        for k in label_keys:
            labels_dict[k].append(sample[k])

    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    for k in label_keys:
        labels_dict[k] = torch.tensor(labels_dict[k], dtype=torch.long)

    return embeddings, labels_dict
