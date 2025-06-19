import argparse
import pickle
import json
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model: RandomForestClassifier = model_dict["model"]
        params = model_dict["params"]
        forest = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            tree_data = {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value.squeeze().tolist()
            }
            forest.append(tree_data)

        model_json = {
            "params": params,
            "classes": model.classes_.tolist(),
            "features": model_dict["features"],
            "model_data": forest
        }
        with open(f"{c}_model.json", "w") as f:
            json.dump(model_json, f)
