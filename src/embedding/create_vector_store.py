import sys
import os
import faiss
import pickle
import numpy as np

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.ingestion.text_cleaning import clean_text
from src.chunking.text_splitter import split_text
from src.embedding.embeddings_model import EmbeddingModel
from src.utils.config import VECTOR_DB_PATH, TEXT_STORE_PATH


def extract_skills(text):
    """Basic skill extraction from text."""
    skills_keywords = [
        "python", "sql", "excel", "tableau", "power bi", "pandas", "numpy",
        "machine learning", "data analysis", "computer vision", "jenkins",
        "deep learning", "nlp", "pytorch", "hugging face", "langchain", "streamlit"
    ]
    text_lower = text.lower()
    return [k for k in skills_keywords if k in text_lower]


def create_vector_store(resume_text):
    """Create FAISS vector DB from uploaded resume"""

    embedder = EmbeddingModel()

    # 1️⃣ Clean resume
    cleaned = clean_text(resume_text)

    # 2️⃣ Chunk resume into smaller pieces
    # Reduce chunk_size to 100 for small resumes to get multiple chunks
    chunks = split_text(cleaned)

    all_chunks = []
    chunk_metadata = []

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_metadata.append({
            "doc_id": 0,
            "skills": extract_skills(chunk),
            "raw_text": chunk
        })

    # 3️⃣ Encode chunks
    embeddings = embedder.encode(all_chunks)

    # 4️⃣ Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # 5️⃣ Save FAISS index and metadata
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_DB_PATH)

    with open(TEXT_STORE_PATH, "wb") as f:
        pickle.dump(chunk_metadata, f)

    print("✅ Resume Vector DB Created")

    # 6️⃣ Wrapper class for similarity search (like older code)
    class VectorStore:
        def __init__(self, index, chunks, embedder):
            self.index = index
            self.chunks = chunks
            self.embedder = embedder

        def similarity_search(self, query, k=5):
            query_vec = self.embedder.encode([query])
            D, I = self.index.search(np.array(query_vec), k)
            results = []
            for i in I[0]:
                class Doc:
                    def __init__(self, text):
                        self.page_content = text
                results.append(Doc(self.chunks[i]))
            return results

    return index, chunk_metadata