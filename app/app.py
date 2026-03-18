import sys
import os
import streamlit as st

# Fix path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.rag_pipeline.rag_system import RAGSystem

st.set_page_config(page_title="RAG Resume Advisor", layout="wide")
st.title("RAG Resume Job Fit Advisor")
st.write("Analyze resumes and get a job fit score with explanation.")

# Initialize RAG system
@st.cache_resource
def load_rag():
    return RAGSystem()

rag = load_rag()

# Input
user_query = st.text_input("Enter the job role or question:")
uploaded_file = st.file_uploader("Upload Resume (.txt)", type=["txt"])

resume_text = None
if uploaded_file:
    resume_text = uploaded_file.read().decode("utf-8")

# Analyze Button
if st.button("Analyze"):
    if resume_text and user_query:
        with st.spinner("Analyzing resume..."):
            result = rag.run(resume_text, user_query)

        st.subheader("Job Fit Score")
        st.metric("Fit Score", f"{result['fit_score']}/100")

        st.subheader("Strengths")
        st.write(result['strengths'])

        st.subheader("Weaknesses")
        st.write(result['weaknesses'])

        st.subheader("Explanation")
        st.write(result['explanation'])

        # ✅ Hide retrieved chunks (optional)
        # st.subheader("Retrieved Resume Chunks")
        # for i, chunk in enumerate(result['retrieved_chunks'], 1):
        #     st.write(f"[Chunk {i}] {chunk[:500]}...")
    else:
        st.warning("Please enter a job role and upload a resume.")