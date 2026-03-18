from src.utils.config import CHUNK_SIZE

def split_text(text):

    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i+CHUNK_SIZE])
        chunks.append(chunk)

    return chunks