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
        models_dict: dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model = model_dict["model"]
        print(
            f"Classification Report for {c} model (Test Set):")
        print(model_dict["report"])
        print(f"params: {model_dict['params']}\n")
        plot_history(c,
                     model_dict["confusion_matrix"],
                     model.classes_)
    plt.tight_layout()
    plt.show()
