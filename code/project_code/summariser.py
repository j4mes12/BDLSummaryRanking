from dataclasses import dataclass
import numpy as np


@dataclass
class RandomSentenceSummariser:
    text: str

    def _generate_summary(self, summary_length=10):
        summary = np.random.choice(
            self.text.split(". "), size=summary_length, replace=False
        )

        summary = ". ".join(summary)

        return summary

    def generate_summaries(self, n=1, summary_length=10):
        return [
            self._generate_summary(summary_length=summary_length)
            for _ in range(n)
        ]
