import os
import argparse

from app.config import settings
from app.services.recommender import RecommenderService
from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.infrastructure.vector_store.faiss_store import FaissVectorStore
from app.infrastructure.preprocessing.factory import make_preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["query", "rebuild", "classify", "train"])
    parser.add_argument("--image", help="Path to image file for query/classify")
    parser.add_argument(
        "--products_dir",
        default="data/products",
        help="Directory of product images for rebuild",
    )
    parser.add_argument(
        "--save_preprocessed",
        action="store_true",
        help="Save preprocessed images to data/preprocessed directory",
    )
    parser.add_argument(
        "--preprocessed_dir",
        default="data/preprocessed",
        help="Directory to save preprocessed images (default: data/preprocessed)",
    )
    parser.add_argument(
        "--category",
        help="Product category for train/classify commands (e.g., shoe, bag)",
    )
    parser.add_argument(
        "--attribute",
        help="Attribute to train (e.g., color, gender, age_group)",
    )
    parser.add_argument(
        "--use-trained",
        action="store_true",
        help="Use trained models for attribute classification (default: zero-shot)",
    )
    args = parser.parse_args()

    # Create preprocessor from config
    preprocessor = make_preprocessor(settings)
    embedding = ClipEmbeddingModel(preprocessor=preprocessor)
    vector_store = FaissVectorStore()
    recommender = RecommenderService(embedding, vector_store)

    if args.command == "rebuild":
        import numpy as np
        import json

        ids = []
        vectors = []
        id_to_filename = {}  # NEW: mapping

        # Create preprocessed directory if saving preprocessed images
        if args.save_preprocessed:
            os.makedirs(args.preprocessed_dir, exist_ok=True)
            print(f"Saving preprocessed images to: {args.preprocessed_dir}")

        for idx, filename in enumerate(os.listdir(args.products_dir)):
            path = os.path.join(args.products_dir, filename)
            print(f"Processing {filename}...")
            embedding_vector = embedding.encode_image(
                path, save_preprocessed=args.save_preprocessed, save_dir=args.preprocessed_dir
            )
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

        # Create preprocessed directory if saving preprocessed images
        if args.save_preprocessed:
            os.makedirs(args.preprocessed_dir, exist_ok=True)
            print(f"Saving preprocessed images to: {args.preprocessed_dir}")

        # Load mapping JSON
        mapping_path = os.path.join(
            os.path.dirname(settings.FAISS_INDEX_PATH), "id_to_filename.json"
        )
        import json

        with open(mapping_path, "r") as f:
            id_to_filename = json.load(f)

        vector_store.load()
        ids, scores = recommender.recommend(
            args.image,
            save_preprocessed=args.save_preprocessed,
            save_dir=args.preprocessed_dir,
        )

        print("\nTop Results:")
        for i, (pid, score) in enumerate(zip(ids, scores)):
            filename = id_to_filename.get(str(pid), "unknown")
            print(
                f"{i + 1}. Product ID: {pid} | Filename: {filename} | Distance: {score:.4f}"
            )

    elif args.command == "train":
        if args.category is None:
            print("Error: --category argument is required for train")
            return

        if args.attribute is None:
            print("Error: --attribute argument is required for train")
            return

        from app.training.train_attribute import train_attribute

        train_attribute(args.category, args.attribute)

    elif args.command == 'classify':
        if args.image is None:
            print("Error: --image argument is required for classify")
            return

        # Category classification (always zero-shot)
        from app.services.category_classifier_service import CategoryClassifierService
        from app.services.product_attribute_service import ProductAttributeService
        from app.services.zero_shot_attribute_service import ZeroShotAttributeService

        category_service = CategoryClassifierService(embedding_model=embedding)
        category, cat_conf = category_service.classify(args.image)
        print(f"Category: {category} (confidence {cat_conf:.2f})")

        # Attribute classification based on category
        if args.use_trained:
            print("\nUsing trained models for attribute classification...")
            attribute_service = ProductAttributeService(embedding_model=embedding)
            try:
                attributes = attribute_service.classify(args.image, category=category)
            except Exception as e:
                print(f"Error loading trained models: {e}")
                print("\nFalling back to zero-shot classification...")
                attribute_service = ZeroShotAttributeService(embedding_model=embedding)
                attributes = attribute_service.classify(args.image, category=category)
        else:
            print("\nUsing zero-shot classification for attributes...")
            attribute_service = ZeroShotAttributeService(embedding_model=embedding)
            attributes = attribute_service.classify(args.image, category=category)

        print("\nAttributes:")
        for attr_name, info in attributes.items():
            print(f" - {attr_name}: {info['value']} (confidence {info['confidence']:.2f})")


if __name__ == "__main__":
    main()
