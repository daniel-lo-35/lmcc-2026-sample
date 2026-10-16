# -*- coding: utf-8 -*-
"""【教師專用 · 參考答案】"""


def build_training_submission(dataset: str, params: dict) -> dict:
    params = dict(params)
    params.update({"epochs": 10, "learningRate": 0.0003, "loraRank": 16, "batchSize": 1, "gradAccum": 4})
    return {"dataset": dataset, "params": params}
