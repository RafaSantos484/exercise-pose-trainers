import argparse
import json
import pickle
from keras.models import model_from_json, Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model_json = model_dict["model_json"]
        weights = model_dict["model_weights"]

        model: Model = model_from_json(model_json)  # type: ignore
        model.set_weights(weights)
        model.save(f"{c}_model.h5")

        model_json = {
            "classes": model_dict["classes"].tolist(),
            "features": model_dict["features"],
            "train_test_split_seed": model_dict["train_test_split_seed"]
        }
        with open(f"{c}_model_info.json", "w") as f:
            json.dump(model_json, f)
