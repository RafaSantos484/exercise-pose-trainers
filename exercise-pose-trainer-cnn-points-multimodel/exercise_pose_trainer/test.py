import os
import argparse
import pickle
import numpy as np
from keras.models import model_from_json

from .utils import get_basename, is_img_file
from .extract_features import CLASSES, extract_features, get_landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path", type=str,
                        help="Path to the test images folder")
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict = pickle.load(f)

    test_path = args.test_path
    img_paths = [path for path in os.listdir(test_path) if is_img_file(path)]

    X = {}
    preds = {}
    for c in CLASSES:
        X[c] = []
        preds[c] = []

    landmarked_img_paths = []
    for img_path in img_paths:
        landmarks = get_landmarks(os.path.join(test_path, img_path))
        if landmarks:
            for c in CLASSES:
                X[c].append(extract_features(landmarks, c))
            landmarked_img_paths.append(img_path)
    img_paths = landmarked_img_paths

    label_encoder = models_dict["label_encoder"]
    for c in CLASSES:
        model_json = models_dict["models"][c]["model_json"]
        weights = models_dict["models"][c]["model_weights"]

        model = model_from_json(model_json)
        model.set_weights(weights)  # type: ignore

        pred_probs = model.predict(X[c], verbose=0)  # type: ignore
        preds[c] = label_encoder.inverse_transform(
            np.argmax(pred_probs, axis=1))

    print("model predictions:")
    for i, img in enumerate(img_paths):
        models_preds = ""
        for c in CLASSES:
            models_preds += f"{c}: {preds[c][i]}, "
        models_preds = models_preds[:-2]
        print(f"{img}: {models_preds}")
