import os
import json
import numpy as np

from cli.message import Message


Msg = Message()

def run_rebuild(embedding, vector_store, products_dir: str = "data/products") -> None:
    """
    Encode all product images in `products_dir`, populate the FAISS index,
    and persist both the index and the id→filename mapping to disk.
    """
    print(Msg.highlight("\nStarted rebuilding process\n"))
    os.makedirs(products_dir, exist_ok=True)

    ids, vectors, id_to_filename = [], [], {}

    for idx, filename in enumerate(os.listdir(products_dir)):
        path = os.path.join(products_dir, filename)
        
        print(Msg.info(f"Processing {filename}..."))
        
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

    print(Msg.highlight("\nIndex rebuilt successfully!"))
