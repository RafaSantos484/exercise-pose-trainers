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

    classes_features = models_dict["classes_features"]
    classes = list(classes_features.keys())
    for c in classes:
        model: RandomForestClassifier = models_dict["models"][c]["model"]
        params = models_dict["models"][c]["params"]
        forest_data = []
        for estimator in model.estimators_:
            tree = estimator.tree_
            tree_data = {
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value.squeeze().tolist()  # shape (n_nodes, n_classes)
            }
            forest_data.append(tree_data)
        # save as json
        model_dict = {
            "params": params,
            "classes": model.classes_.tolist(),
            "forest": forest_data
        }
        with open(f"{c}_model.json", "w") as f:
            json.dump(model_dict, f)
