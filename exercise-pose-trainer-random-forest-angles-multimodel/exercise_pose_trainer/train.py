import argparse
import pickle

from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
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

    models_dict = {}
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
            "n_estimators": [2, 5, 10, 30, 50, 80, 100],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 20, 40, 60, 80, 100, 150, 200, 300],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [1, 2, 3, 4],
            "bootstrap": [True, False]
        }
        model = RandomForestClassifier()
        grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        model_classes = best_model.classes_
        params = grid_search.best_params_
        params["train_test_split_seed"] = seed

        y_pred = best_model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=model_classes, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        models_dict[c] = {
            "model": best_model,
            "params": params,
            "report": report,
            "confusion_matrix": cm,
            "features": classes_features[c]
        }

    for c in classes:
        print(f"Classification Report for {c} model(Test Set):")
        print(models_dict[c]["report"])
        print(f"params: {models_dict[c]['params']}\n")

    model_name = get_basename(base_path)
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(models_dict, f)

    if args.plot:
        for c in classes:
            plot_history(c,
                         models_dict[c]["confusion_matrix"],
                         models_dict[c]["model"].classes_)
        plt.tight_layout()
        plt.show()
