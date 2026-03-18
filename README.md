# AI Resume Intelligence System
RAG-Based Job Fit Analyzer
## Project Overview

Recruiters often spend significant time manually reviewing resumes and comparing them with job descriptions to determine candidate suitability. This process is time-consuming, inconsistent, and difficult to scale when dealing with large volumes of applications.

This project presents an AI-powered Resume Intelligence System that automatically analyzes resumes and evaluates how well a candidate fits a specific job role using Retrieval-Augmented Generation (RAG).

The system leverages vector search, transformer embeddings, and large language models to retrieve relevant resume content and generate a job fit score with explanation.
## Problem Statement

Modern hiring pipelines receive thousands of resumes for each job posting, making manual evaluation inefficient and prone to bias.

Traditional keyword-based resume screening systems often fail to capture semantic similarity between candidate skills and job requirements.

The goal of this project is to build an intelligent resume analysis system that:

Understands resume content

Matches resumes with job descriptions

Retrieves relevant resume segments

Generates a Job Fit Score (0–100)

Provides a clear explanation for the score

This system helps automate early-stage candidate screening using LLM-powered reasoning.

## Project Objectives

Build an AI system to analyze resumes automatically

Implement Retrieval-Augmented Generation (RAG) architecture

Convert resume text into vector embeddings

Store vectors in a FAISS vector database

Retrieve relevant resume chunks using semantic search

Generate job fit scoring and explanation using an LLM

Build an interactive Streamlit interface

## System Architecture

The project follows a Retrieval-Augmented Generation (RAG) pipeline:

Resume + Job Description
        │
        ▼
Text Cleaning
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
(sentence-transformers)
        │
        ▼
Vector Storage
(FAISS Index)
        │
        ▼
Semantic Retrieval
        │
        ▼
LLM Reasoning
(FLAN-T5)
        │
        ▼
Job Fit Score + Explanation
## Project Folder Structure
project_root
│
├── app
│   ├── app.py
│   └── interface.py
│
├── data
│   ├── raw
│   │   ├── resumes
│   │   └── job_descriptions
│   │
│   ├── processed
│   │   ├── cleaned_resumes
│   │   └── cleaned_jd
│   │
│   └── embeddings
│       └── vector_store
│
├── models
│   ├── saved_models
│   └── tokenizers
│
├── notebooks
│   └── test_drive.ipynb
│
├── outputs
│   ├── logs
│   └── results
│
├── src
│   ├── chunking
│   ├── embedding
│   ├── ingestion
│   ├── llm
│   ├── rag_pipeline
│   ├── retriever
│   └── utils
│
├── tests
│   ├── test_dataset.py
│   └── test_retriever.py
│
├── README.md
└── requirements.txt
## Technologies Used
Category	Tools
Programming	Python
Embeddings	Sentence Transformers
LLM	FLAN-T5
Vector Database	FAISS
NLP	HuggingFace Transformers
Dataset	HuggingFace Datasets
UI	Streamlit
PDF Parsing	pdfplumber
ML Utilities	Scikit-learn, NumPy, Pandas
## Dataset

Dataset used:

Resume Screening Dataset

It contains resume and job description pairs separated by [sep].

The project processes 1500 samples to ensure compatibility with laptop environments.

## Key Features

✔ Resume semantic understanding
✔ Job description matching
✔ Retrieval-Augmented Generation (RAG)
✔ Vector similarity search using FAISS
✔ Job fit scoring (0–100)
✔ LLM-generated explanation
✔ Streamlit interactive interface
✔ Modular pipeline architecture

## Output Example
Job Role: Data Analyst

Job Fit Score: 75/100

Explanation:
The candidate demonstrates strong experience in Python, data analysis, and visualization tools such as Tableau. The resume highlights relevant analytical projects and familiarity with machine learning techniques. However, the candidate lacks explicit experience in advanced business intelligence tools required for the role. Overall, the candidate shows a strong alignment with the core analytical skills required for the job.
## Future Improvements

Named Entity Recognition for skill extraction

Resume ranking system

Multi-resume comparison

Better scoring model using ML

LLM fine-tuning

Deployment using Docker