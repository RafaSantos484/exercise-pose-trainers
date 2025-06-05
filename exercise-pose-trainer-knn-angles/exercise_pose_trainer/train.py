import argparse
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from exercise_pose_trainer.utils import get_basename

from .extract_features import load_features


def plot_history(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(colorbar=False)
    disp.ax_.set_title("KNN Confusion Matrix")
    plt.show()


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
    X, y = load_features(base_path)
    print("Loaded features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed)

    print("Training model...")
    # For p=1, minkowski = manhattan
    # For p=2, minkowski = euclidean
    param_grid = {
        "n_neighbors": list(range(1, 20, 2)),
        "p": list(range(1, 11)),
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"]
    }
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Finished training model")
    print(
        f"Best params: {grid_search.best_params_}")
    print(
        f"Best score: {grid_search.best_score_}\n")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Classification Report (Test Set):")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    if args.plot:
        plot_history(y_test, y_pred, best_model.classes_)

    cm = confusion_matrix(y_test, y_pred)
    model_data = {"model": grid_search.best_estimator_,
                  "params": grid_search.best_params_,
                  "classification_report": report,
                  "confusion_matrix": cm,
                  "classes": best_model.classes_
                  }

    model_name = get_basename(base_path)
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
