import argparse
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def plot_history(model_name: str, confusion_matrix, classes):
    # Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=classes)
    disp.plot(colorbar=False)
    disp.ax_.set_title(f"{model_name} Confusion Matrix")

    # plt.tight_layout()
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict = pickle.load(f)

    classes_features = models_dict["classes_features"]
    classes = list(classes_features.keys())
    for c in classes:
        print(
            f"Classification Report for {c} model (Test Set):")
        print(models_dict["models"][c]["report"])
        print(f"params: {models_dict['models'][c]['params']}\n")
        plot_history(c,
                     models_dict["models"][c]["confusion_matrix"],
                     ["incorrect", "correct"])
    plt.tight_layout()
    plt.show()
