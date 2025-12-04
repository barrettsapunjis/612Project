from sentalyzer import (
    SVMABSAModel,
    SetFitABSAModel,
    concat_aspect,
    extract_all_org_aspects_batch,
    load_sentfin_df,
    ABSASample,
)


from sentalyzer.realtime.yahoo_scraper import fetch_news_for_ticker, toSample
from sentalyzer.data.samples import ABSASample
from sentalyzer.extraction.api import extract_all_org_aspects


SVM_MODEL_DIR = "models/svm_sentfin-BEST"
SETFIT_MODEL_DIR = "models/setfit-absa-sentfin-BEST"
def main():
    news = fetch_news_for_ticker("AAPL", fetch_full_text=False)
    samples = toSample(news)
    new_samples = []
    for sample in samples:
        aspects = extract_all_org_aspects(sample.text, min_score=0.7)
        for aspect in aspects:
            new_samples.append(ABSASample(text=sample.text, aspect=aspect.aspect, label=None))
    model = SVMABSAModel.from_dir(str(SVM_MODEL_DIR), combine_fn=concat_aspect)
    model2 = SetFitABSAModel.from_dir(str(SETFIT_MODEL_DIR), combine_fn=concat_aspect)


    preds = model.predict_samples(new_samples)
    scores = model.predict_scores(new_samples)
    preds2 = model2.predict_samples(new_samples)
    scores2 = model2.predict_scores(new_samples)
    i = 0
    for sample, pred, pred2 in zip(new_samples, preds, preds2):
        print(f"Sample: {sample.text}")
        print(f"Aspect: {sample.aspect}")
        print(f"Pred SVM: {pred}")
        print(f"Scores SVM: {scores[i]}")
        print(f"Pred SetFit: {pred2}")
        print(f"Scores SetFit: {scores2[i]}")
        print("-" * 40)
        i += 1

if __name__ == "__main__":
    main()