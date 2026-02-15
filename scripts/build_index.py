import os
import numpy as np

from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.infrastructure.vector_store.faiss_store import FaissVectorStore


PRODUCTS_PATH = "data/products"

def main():
    model = ClipEmbeddingModel()
    store = FaissVectorStore()

    ids = []
    vectors = []

    for idx, filename in enumerate(os.listdir(PRODUCTS_PATH)):
        path = os.path.join(PRODUCTS_PATH, filename)

        print(f"Processing {filename}")
        embedding = model.encode_image(path)

        ids.append(idx)
        vectors.append(embedding)

    vectors = np.array(vectors).astype("float32")

    store.add(ids, vectors)
    store.save()

    print("Index built successfully!")

if __name__ == "__main__":
    main()
