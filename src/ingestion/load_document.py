from datasets import load_dataset
from src.utils.config import DATASET_NAME, MAX_SAMPLES

def load_document():

    dataset = load_dataset(DATASET_NAME, split="train")

    texts = dataset["text"][:MAX_SAMPLES]

    documents = []

    for t in texts:
        resume, job = t.split("[sep]")
        documents.append({
            "resume": resume.strip(),
            "job_description": job.strip()
        })

    return documents