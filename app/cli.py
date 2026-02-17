import os
import argparse
import shlex
import json

from app.config import settings
from app.services.recommender import RecommenderService
from app.infrastructure.embedding.clip_model import ClipEmbeddingModel
from app.infrastructure.vector_store.faiss_store import FaissVectorStore
from app.infrastructure.preprocessing.factory import make_preprocessor
from app.infrastructure.cache.memory_cache import MemoryCache


def parse_command(command_str):
    """
    Parse interactive command string into argparse-like namespace.
    """
    parts = shlex.split(command_str)
    cmd_args = {
        "command": None,
        "image": None,
        "products_dir": None,
        "save_preprocessed": False,
        "preprocessed_dir": None,
        "category": None,
        "attribute": None,
        "use_trained": False,
        "clear": False,
        "list": False,
    }

    if not parts:
        return None

    cmd_args["command"] = parts[0]

    i = 1
    while i < len(parts):
        p = parts[i]

        # Flags with values
        if p == "--image" and i + 1 < len(parts):
            cmd_args["image"] = parts[i + 1]
            i += 2
        elif p == "--products_dir" and i + 1 < len(parts):
            cmd_args["products_dir"] = parts[i + 1]
            i += 2
        elif p == "--preprocessed_dir" and i + 1 < len(parts):
            cmd_args["preprocessed_dir"] = parts[i + 1]
            i += 2
        elif p == "--category" and i + 1 < len(parts):
            cmd_args["category"] = parts[i + 1]
            i += 2
        elif p == "--attribute" and i + 1 < len(parts):
            cmd_args["attribute"] = parts[i + 1]
            i += 2

        # Flags without values
        elif p == "--save-preprocessed":
            cmd_args["save_preprocessed"] = True
            i += 1
        elif p == "--use-trained":
            cmd_args["use_trained"] = True
            i += 1
        elif p == "--clear":
            cmd_args["clear"] = True
            i += 1
        else:
            # Check if it's a cache subcommand
            if cmd_args["command"] == "cache":
                if p == "list":
                    cmd_args["list"] = True
                    i += 1
                elif p == "clear":
                    cmd_args["clear"] = True
                    i += 1
                else:
                    print(f"Unknown cache subcommand: {p}")
                    i += 1
            else:
                print(f"Unknown argument: {p}")
                i += 1

    return cmd_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve", "rebuild", "train"])
    parser.add_argument("--products_dir", default="data/products", help="Directory of product images for rebuild")
    parser.add_argument("--category", help="Category for training")
    parser.add_argument("--attribute", help="Attribute for training")
    args = parser.parse_args()

    preprocessor = make_preprocessor(settings)
    embedding = ClipEmbeddingModel(preprocessor=preprocessor)
    vector_store = FaissVectorStore()
    recommender = RecommenderService(embedding, vector_store)
    cache = MemoryCache()

    # ---------- Non-interactive rebuild ----------
    if args.command == "rebuild":
        import numpy as np

        ids, vectors, id_to_filename = [], [], {}
        os.makedirs(args.products_dir, exist_ok=True)
        for idx, filename in enumerate(os.listdir(args.products_dir)):
            path = os.path.join(args.products_dir, filename)
            print(f"Processing {filename}...")
            vector = embedding.encode_image(path)
            ids.append(idx)
            vectors.append(vector)
            id_to_filename[idx] = filename

        vectors = np.array(vectors).astype("float32")
        vector_store.add(ids, vectors)
        vector_store.save()

        mapping_path = os.path.join(os.path.dirname(vector_store.index_path), "id_to_filename.json")
        with open(mapping_path, "w") as f:
            json.dump(id_to_filename, f, indent=2)

        print("Index rebuilt successfully!")
        return

    # ---------- Non-interactive train ----------
    if args.command == "train":
        if not args.category or not args.attribute:
            print("Please provide --category and --attribute for training")
            return
        from app.training.train_attribute import train_attribute

        train_attribute(args.category, args.attribute)
        return

    # ---------- Interactive serve ----------
    if args.command == "serve":
        print("Entering interactive serve mode. Type 'exit' to quit.")
        while True:
            try:
                command_str = input("\n>>> ").strip()
                if command_str.lower() in ["exit", "quit"]:
                    print("Exiting serve...")
                    break

                cmd = parse_command(command_str)
                if not cmd:
                    continue

                # ---------- REBUILD ----------
                if cmd["command"] == "rebuild":
                    import numpy as np

                    products_dir = cmd["products_dir"] or "data/products"
                    ids, vectors, id_to_filename = [], [], {}
                    for idx, filename in enumerate(os.listdir(products_dir)):
                        path = os.path.join(products_dir, filename)
                        print(f"Processing {filename}...")
                        vector = embedding.encode_image(path)
                        ids.append(idx)
                        vectors.append(vector)
                        id_to_filename[idx] = filename

                    vectors = np.array(vectors).astype("float32")
                    vector_store.add(ids, vectors)
                    vector_store.save()

                    mapping_path = os.path.join(os.path.dirname(vector_store.index_path), "id_to_filename.json")
                    with open(mapping_path, "w") as f:
                        json.dump(id_to_filename, f, indent=2)

                    print("Index rebuilt successfully!")

                # ---------- QUERY ----------
                elif cmd["command"] == "query":
                    img_path = cmd["image"]
                    if not img_path or not os.path.exists(img_path):
                        print("Please provide valid --image for query")
                        continue

                    if not os.path.exists(settings.FAISS_INDEX_PATH):
                        print("FAISS index not found. Rebuild index first.")
                        continue

                    vector_store.load()
                    ids, scores = recommender.recommend(img_path)
                    mapping_path = os.path.join(os.path.dirname(settings.FAISS_INDEX_PATH), "id_to_filename.json")
                    with open(mapping_path, "r") as f:
                        id_to_filename = json.load(f)

                    print("\nTop Results:")
                    for i, (pid, score) in enumerate(zip(ids, scores)):
                        filename = id_to_filename.get(str(pid), "unknown")
                        print(f"{i + 1}. Product ID: {pid} | Filename: {filename} | Distance: {score:.4f}")

                # ---------- CLASSIFY ----------
                elif cmd["command"] == "classify":
                    img_path = cmd["image"]
                    use_trained = cmd["use_trained"]

                    if not img_path or not os.path.exists(img_path):
                        print("Please provide valid --image for classify")
                        continue

                    from app.services.category_classifier_service import CategoryClassifierService
                    from app.services.product_attribute_service import ProductAttributeService
                    from app.services.zero_shot_attribute_service import ZeroShotAttributeService

                    category_service = CategoryClassifierService(embedding_model=embedding)
                    category, cat_conf = category_service.classify(img_path)
                    print(f"Category: {category} (confidence {cat_conf:.2f})")

                    if use_trained:
                        print("\nUsing trained models for attribute classification...")
                        attribute_service = ProductAttributeService(embedding_model=embedding, cache=cache)
                        try:
                            attributes = attribute_service.classify(img_path, category=category)
                        except Exception as e:
                            print(f"Error loading trained models: {e}")
                            print("\nFalling back to zero-shot classification...")
                            attribute_service = ZeroShotAttributeService(embedding_model=embedding)
                            attributes = attribute_service.classify(img_path, category=category)
                    else:
                        print("\nUsing zero-shot classification for attributes...")
                        attribute_service = ZeroShotAttributeService(embedding_model=embedding)
                        attributes = attribute_service.classify(img_path, category=category)

                    print("\nAttributes:")
                    for attr_name, info in attributes.items():
                        print(f" - {attr_name}: {info['value']} (confidence {info['confidence']:.2f})")

                # ---------- CACHE ----------
                elif cmd["command"] == "cache":
                    if cmd["clear"]:
                        cache.clear()
                        print("All caches cleared")
                    elif cmd["list"]:
                        keys = cache.keys()
                        print(f"Listing {len(keys)} keys:")
                        for k in keys:
                            print(" -", k)
                    else:
                        print("MemoryCache is live and will speed up repeated queries/classifications")

                else:
                    print(f"Unknown command: {cmd['command']}")

            except KeyboardInterrupt:
                print("\nExiting serve...")
                break


if __name__ == "__main__":
    main()
