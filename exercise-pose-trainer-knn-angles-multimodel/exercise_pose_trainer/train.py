import argparse
import pickle

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from .model import plot_history
from .extract_features import load_features
from .utils import get_basename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the features folder")
    parser.add_argument("--seed", type=int,
                        help="Seed used for reproducibility")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the training history")
    args = parser.parse_args()

    base_path = args.path
    seed = args.seed

    print("Loading features...")
    X_classes, y_classes, classes_features = load_features(base_path)
    print("Loaded features")

    models_dict = {"models": {}, "classes_features": classes_features}
    classes = list(classes_features.keys())
    for c in classes:
        X = np.array(X_classes[c])
        y = np.array(y_classes[c])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed)

        print(f"Training {c} model...")
        # For p=1, minkowski = manhattan
        # For p=2, minkowski = euclidean
        param_grid = {
            "n_neighbors": list(range(1, 20, 2)),
            "p": list(range(1, 21)),
            "weights": ["uniform", "distance"],
            "metric": ["minkowski"]
        }
        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        model_classes = best_model.classes_
        params = grid_search.best_params_

        y_pred = best_model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=model_classes, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        models_dict["models"][c] = {
            "model": best_model,
            "params": params,
            "report": report,
            "confusion_matrix": cm,
        }

    for c in classes:
        print(f"Classification Report for {c} model(Test Set):")
        print(models_dict["models"][c]["report"])
        print(f"params: {models_dict['models'][c]['params']}\n")

    model_name = get_basename(base_path)
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(models_dict, f)

    if args.plot:
        for c in classes:
            plot_history(c,
                         models_dict["models"][c]["confusion_matrix"],
                         models_dict["models"][c]["model"].classes_)
        plt.tight_layout()
        plt.show()
