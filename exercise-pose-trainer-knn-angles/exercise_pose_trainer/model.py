import argparse
import pickle

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def report_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        model_dict = pickle.load(f)

    print(model_dict["classification_report"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=model_dict["confusion_matrix"], display_labels=model_dict["classes"])
    disp.plot(colorbar=False)
    disp.ax_.set_title("KNN Confusion Matrix")

    plt.tight_layout()
    plt.show()
