from transformers import pipeline
from src.utils.config import LLM_MODEL


class LLMPipeline:
    """
    LLM wrapper for text generation with proper sampling.
    """

    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model=LLM_MODEL
        )

    def generate(self, prompt):
        output = self.generator(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        return output[0]["generated_text"]