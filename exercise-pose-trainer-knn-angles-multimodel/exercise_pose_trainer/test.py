import os
import argparse
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .utils import get_basename, is_img_file
from .extract_features import extract_features, get_landmarks


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

    classes_features = models_dict["classes_features"]
    classes = list(classes_features.keys())
    X = {}
    preds = {}
    for c in classes:
        X[c] = []
        preds[c] = []

    landmarked_img_paths = []
    for img_path in img_paths:
        landmarks = get_landmarks(os.path.join(test_path, img_path))
        if landmarks:
            for c in classes:
                X[c].append(extract_features(
                    landmarks, classes_features[c]["angles"]))
            landmarked_img_paths.append(img_path)
    img_paths = landmarked_img_paths

    for c in classes:
        model: KNeighborsClassifier = models_dict["models"][c]["model"]
        preds[c] = model.predict(np.array(X[c]))

    print("model predictions:")
    for i, img in enumerate(img_paths):
        models_preds = ""
        for c in classes:
            models_preds += f"{c}: {preds[c][i]}, "
        models_preds = models_preds[:-2]
        print(f"{img}: {models_preds}")
