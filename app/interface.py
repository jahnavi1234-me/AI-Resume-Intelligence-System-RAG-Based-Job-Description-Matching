# app/interface.py
import sys
import os

# Add project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.rag_pipeline.rag_system import RAGSystem
import re

def extract_score(response_text):
    match = re.search(r"Job Fit Score: (\d+)/100", response_text)
    if match:
        return int(match.group(1))
    return None

def main():
    rag = RAGSystem()
    print("\n=== Resume Job Fit Advisor ===")  # Should show immediately

    while True:
        query = input("\nEnter job-role query (or 'quit'): ")
        if query.lower() == "quit":
            break
        response = rag.run(query)
        print("\n=== Analysis ===")
        print(response)
        score = extract_score(response)
        print(f"\nExtracted Score: {score}/100")

if __name__ == "__main__":
    main()
result = rag.run(query)
print(f"Job Fit Score: {result['fit_score']}/100")
print("Explanation:")
print(result['explanation'])