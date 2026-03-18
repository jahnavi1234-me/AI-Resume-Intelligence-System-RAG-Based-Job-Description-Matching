# src/ingestion/pdf_loader.py

import os
import pdfplumber

def load_pdf(file_path: str) -> str:
    """
    Extract text from a single PDF file.
    """
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text.strip()


def load_all_pdfs(folder_path: str) -> list:
    """
    Load all PDFs from a folder.
    """
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            text = load_pdf(full_path)
            documents.append(text)

    return documents