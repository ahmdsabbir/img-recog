import os
import argparse

from app.config import settings
from app.services.recommender import RecommenderService
from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.infrastructure.vector_store.faiss_store import FaissVectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["query", "rebuild"])
    parser.add_argument("--image", help="Path to image file for query")
    parser.add_argument(
        "--products_dir",
        default="data/products",
        help="Directory of product images for rebuild",
    )
    args = parser.parse_args()

    embedding = ClipEmbeddingModel()
    vector_store = FaissVectorStore()
    recommender = RecommenderService(embedding, vector_store)

    if args.command == "rebuild":
        import numpy as np
        import json

        ids = []
        vectors = []
        id_to_filename = {}  # NEW: mapping

        for idx, filename in enumerate(os.listdir(args.products_dir)):
            path = os.path.join(args.products_dir, filename)
            print(f"Processing {filename}...")
            embedding_vector = embedding.encode_image(path)
            ids.append(idx)
            vectors.append(embedding_vector)

            id_to_filename[idx] = filename  # store mapping

        vectors = np.array(vectors).astype("float32")
        vector_store.add(ids, vectors)
        vector_store.save()

        # save mapping JSON
        os.makedirs(os.path.dirname(vector_store.index_path), exist_ok=True)
        mapping_path = os.path.join(
            os.path.dirname(vector_store.index_path), "id_to_filename.json"
        )
        with open(mapping_path, "w") as f:
            json.dump(id_to_filename, f, indent=2)

        print("Index rebuilt successfully!")

    elif args.command == "query":
        if args.image is None:
            print("Error: --image argument is required for query")
            return

        if not os.path.exists(settings.FAISS_INDEX_PATH):
            print("FAISS index not found. Rebuild index first using 'rebuild' command.")
            return

        # Load mapping JSON
        mapping_path = os.path.join(
            os.path.dirname(settings.FAISS_INDEX_PATH), "id_to_filename.json"
        )
        import json

        with open(mapping_path, "r") as f:
            id_to_filename = json.load(f)

        vector_store.load()
        ids, scores = recommender.recommend(args.image)

        print("\nTop Results:")
        for i, (pid, score) in enumerate(zip(ids, scores)):
            filename = id_to_filename.get(str(pid), "unknown")
            print(
                f"{i + 1}. Product ID: {pid} | Filename: {filename} | Distance: {score:.4f}"
            )


if __name__ == "__main__":
    main()
