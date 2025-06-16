import argparse
import pickle
from keras.models import model_from_json, Model
# import tensorflowjs as tfjs


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
        model_json = models_dict["models"][c]["model_json"]
        weights = models_dict["models"][c]["model_weights"]

        model: Model = model_from_json(model_json)  # type: ignore
        model.set_weights(weights)
        model.save(f"{c}_model.h5")
        # tfjs.converters.save_keras_model(model, f"{c}_model")
