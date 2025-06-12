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
        models_dict = pickle.load(f)

    classes_features = models_dict["classes_features"]
    classes = list(classes_features.keys())
    for c in classes:
        model: KNeighborsClassifier = models_dict["models"][c]["model"]
        params = models_dict["models"][c]["params"]
        # save as json
        model_dict = {
            "params": params,
            "classes": model.classes_.tolist(),
            "X": model._fit_X.tolist(),
            "y": model._y.tolist(),
        }
        with open(f"{c}_model.json", "w") as f:
            json.dump(model_dict, f)
