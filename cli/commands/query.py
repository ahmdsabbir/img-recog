import os
import json

from app.config import settings
from cli.message import Message


Msg = Message()

def run_query(recommender, vector_store, img_path: str) -> None:
    """
    Load the FAISS index and print the top similar products for `img_path`.
    """
    if not os.path.exists(settings.FAISS_INDEX_PATH):
        print(Msg.alert("FAISS index not found. Rebuild index first."))
        return

    vector_store.load()
    ids, scores = recommender.recommend(img_path)

    mapping_path = os.path.join(
        os.path.dirname(settings.FAISS_INDEX_PATH), "id_to_filename.json"
    )
    with open(mapping_path, "r") as f:
        id_to_filename = json.load(f)

    print(Msg.info("\nTop Results:"))
    for i, (pid, score) in enumerate(zip(ids, scores)):
        filename = id_to_filename.get(str(pid), "unknown")
        print(
            f"{i + 1}. Product ID: {pid} | Filename: {filename} | Distance: {score:.4f}"
        )
