import argparse
import pickle
import json
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict: dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model: LogisticRegression = model_dict["model"]
        model_json = {
            "params": model_dict["params"],
            "classes": model.classes_.tolist(),
            "features": model_dict["features"],
            "model_data": {
                "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist()
            }
        }
        with open(f"{c}_model.json", "w") as f:
            json.dump(model_json, f)
