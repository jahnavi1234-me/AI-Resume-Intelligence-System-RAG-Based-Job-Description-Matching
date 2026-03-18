DATASET_NAME = "scmlewis/Resume_Screening_Data_Classification"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"

VECTOR_DB_PATH = "data/embeddings/vector_store/faiss_index/index.faiss"
TEXT_STORE_PATH = "data/embeddings/vector_store/faiss_index/texts.pkl"

CHUNK_SIZE = 300
TOP_K = 3
MAX_SAMPLES = 1500   # safe for laptop