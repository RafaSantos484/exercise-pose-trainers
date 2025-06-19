import argparse
import pickle
import json
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict = pickle.load(f)

    for classname, model_dict in models_dict.items():
        model: SVC = model_dict["model"]
        model_export = {
            "params": model_dict["params"],
            "features": model_dict["features"],
            "classes": model.classes_.tolist(),
            "model_data": {
                "kernel": model.kernel,
                "support_vectors": model.support_vectors_.tolist(),
                "dual_coef": model.dual_coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "gamma": model._gamma,  # pode ser float ou 'scale'/'auto'
                "coef0": model.coef0,
                "degree": model.degree,
            }
        }

        with open(f"{classname}_model.json", "w") as f:
            json.dump(model_export, f)
