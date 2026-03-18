from src.llm.llm_pipeline import LLMPipeline
from src.embedding.embeddings_model import EmbeddingModel
from src.embedding.create_vector_store import create_vector_store
import numpy as np
import re

class RAGSystem:
    """
    RAG system that analyzes an uploaded resume
    and computes job fit score for a given role using semantic similarity.
    """

    def __init__(self):
        self.llm = LLMPipeline()
        self.embedding_model = EmbeddingModel()

    def run(self, resume_text, query):
        # 1️⃣ Create vector DB from resume
        index, metadata = create_vector_store(resume_text)

        # 2️⃣ Encode query and retrieved chunks
        retrieved_chunks = [chunk["raw_text"] for chunk in metadata]
        chunk_embeddings = self.embedding_model.encode(retrieved_chunks)
        query_embedding = self.embedding_model.encode([query])[0]

        # 3️⃣ Compute semantic similarity between query and chunks
        similarities = []
        for emb in chunk_embeddings:
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append(sim)

        # 4️⃣ Top-K chunks for explanation
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_chunks = [retrieved_chunks[i] for i in top_indices]

        # 5️⃣ Compute fit score: average similarity of top-K chunks
        avg_similarity = np.mean([similarities[i] for i in top_indices])
        fit_score = int(avg_similarity * 100)
        fit_score = min(fit_score, 95)  # cap max at 95

        # 6️⃣ Determine strengths & weaknesses (words in query that appear in top chunks)
        query_skills = re.findall(r'\w+', query.lower())
        strengths = []
        weaknesses = []
        for skill in query_skills:
            matched = any(skill in chunk.lower() for chunk in top_chunks)
            if matched:
                strengths.append(skill)
            else:
                weaknesses.append(skill)

        # 7️⃣ Prepare text for LLM explanation
        all_chunks_text = " ".join(top_chunks)
        all_chunks_text = " ".join(all_chunks_text.split()[:2000])

        prompt = f"""
Given the resume chunks below:

{all_chunks_text}

Job Role: {query}

The job fit score is {fit_score}/100.

 
"""
        explanation = self.llm.generate(prompt).strip()

        return {
            "fit_score": fit_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "explanation": explanation,
            
        }