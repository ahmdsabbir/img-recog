import os
import json

import numpy as np


def run_rebuild(embedding, vector_store, products_dir: str = "data/products") -> None:
    """
    Encode all product images in `products_dir`, populate the FAISS index,
    and persist both the index and the id→filename mapping to disk.
    """
    os.makedirs(products_dir, exist_ok=True)

    ids, vectors, id_to_filename = [], [], {}

    for idx, filename in enumerate(os.listdir(products_dir)):
        path = os.path.join(products_dir, filename)
        print(f"Processing {filename}...")
        vector = embedding.encode_image(path)
        ids.append(idx)
        vectors.append(vector)
        id_to_filename[idx] = filename

    vector_store.add(ids, np.array(vectors).astype("float32"))
    vector_store.save()

    mapping_path = os.path.join(
        os.path.dirname(vector_store.index_path), "id_to_filename.json"
    )
    with open(mapping_path, "w") as f:
        json.dump(id_to_filename, f, indent=2)

    print("Index rebuilt successfully!")
