import pandas as pd
from utils.utils import analyze_sentiment


def process_batch(batch_data, lr_model, bow):
    predictions = []
    polarities = []
    emotions_list = []
    for text in batch_data["text"]:
        pred, pol, emo = analyze_sentiment(text, lr_model, bow)
        predictions.append(pred)
        polarities.append(pol)
        emotions_list.append(emo)

    batch_data["predicted_sentiment"] = predictions
    batch_data["polarity"] = polarities
    batch_data["emotions"] = emotions_list

    return batch_data
