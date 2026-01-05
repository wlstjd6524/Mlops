import pickle
from typing import Union

import pandas as pd


def load_model_bundle(model_path: str):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["preprocess"]


def predict(model, preprocess, data: Union[pd.DataFrame, dict, list]):
    """
    data:
      - DataFrame
      - dict (single sample)
      - list[dict] (batch)
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Unsupported input type. Use DataFrame / dict / list[dict].")

    X = preprocess.transform(df)
    preds = model.predict(X)
    return preds
