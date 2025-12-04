from typing import List, Dict

from ner_extractor import NERExtractor
from ticker_mapping import map_company_to_ticker
from absa_model import ABSAModel


class MultiEntityABSA:
    def __init__(self):
        self.ner = NERExtractor()
        self.absa = ABSAModel()

    def analyze_article(self, article_text: str) -> List[Dict]:
        companies = self.ner.extract_companies(article_text)
        results = []

        for company in companies:
            ticker = map_company_to_ticker(company)
            if not ticker:
                continue

            absa_result = self.absa.predict(article_text, aspect=company)

            results.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "sentiment": absa_result["label"],
                    "raw_output": absa_result["raw"],
                }
            )

        return results
