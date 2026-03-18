# test_retriever.py

from src.retriever.retriever import Retriever

# Initialize retriever
retriever = Retriever()

# Test query
query = "Data Analyst with Python experience"

# Retrieve chunks
results = retriever.retrieve(query)

print("Retrieved chunks:", results)
print("Number of chunks:", len(results))