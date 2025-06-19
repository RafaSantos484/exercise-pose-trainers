import argparse
import pickle
import json
from sklearn.neighbors import KNeighborsClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict: dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model: KNeighborsClassifier = model_dict["model"]
        model_json = {
            "params": model_dict["params"],
            "classes": model.classes_.tolist(),
            "features": model_dict["features"],
            "model_data": {
                "X": model._fit_X.tolist(),
                "y": model._y.tolist(),
            }
        }
        with open(f"{c}_model.json", "w") as f:
            json.dump(model_json, f)
