import faiss
import pickle
import numpy as np
import os

from src.ingestion.load_document import load_document
from src.ingestion.text_cleaning import clean_text
from src.chunking.text_splitter import split_text
from src.embedding.embeddings_model import EmbeddingModel
from src.utils.config import VECTOR_DB_PATH, TEXT_STORE_PATH


def create_vector_store():

    docs = load_document()
    embedder = EmbeddingModel()

    all_chunks = []

    for doc in docs:
        combined = doc["resume"] + " " + doc["job_description"]
        cleaned = clean_text(combined)
        chunks = split_text(cleaned)
        all_chunks.extend(chunks)

    embeddings = embedder.encode(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(TEXT_STORE_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Vector DB Created")


# 👇 THIS MUST BE AT LEFT MOST POSITION
if __name__ == "__main__":
    create_vector_store()