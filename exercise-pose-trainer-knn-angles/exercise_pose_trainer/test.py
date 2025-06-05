import os
import argparse
import pickle
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
        model_dict = pickle.load(f)

    test_path = args.test_path
    img_paths = [path for path in os.listdir(test_path) if is_img_file(path)]

    X = []
    landmarked_img_paths = []
    for img_path in img_paths:
        landmarks = get_landmarks(os.path.join(test_path, img_path))
        if landmarks:
            X.append(extract_features(landmarks))
            landmarked_img_paths.append(img_path)
    img_paths = landmarked_img_paths

    model: KNeighborsClassifier = model_dict["model"]
    predictions = model.predict(X)  # type: ignore
    print("model predictions:")
    for img, pred in zip(img_paths, predictions):
        print(f"{img}: {pred}")
