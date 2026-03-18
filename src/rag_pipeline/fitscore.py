import re

class FitScorer:
    def extract_score(self, text):
        match = re.search(r"Job Fit Score: (\d+)/100", text)
        if match:
            return int(match.group(1))
        return None