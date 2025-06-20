import argparse
import pickle

from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical

from .model import get_model, plot_history
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

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_onehot = to_categorical(y_encoded)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.3, random_state=seed)

        input_shape = X.shape[1:]
        labels = label_encoder.classes_
        num_classes = len(labels)
        model = get_model(input_shape, num_classes=num_classes)
        print(f"Training {c} model...")
        early_stopping_callback = EarlyStopping(
            patience=200, restore_best_weights=True)
        reduce_lr_callback = ReduceLROnPlateau(patience=50)
        history = model.fit(X_train, y_train,
                            epochs=10000,
                            # epochs=100,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping_callback,
                                       reduce_lr_callback],
                            verbose=1)  # type: ignore

        y_pred = model.predict(X_test, verbose=0)  # type: ignore
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)

        report = classification_report(
            y_test_labels, y_pred_labels, target_names=labels, digits=4)
        cm = confusion_matrix(y_test_labels, y_pred_labels)
        models_dict[c] = {
            "model_json": model.to_json(),
            "model_weights": model.get_weights(),
            "report": report,
            "confusion_matrix": cm,
            "history": history,
            "classes": labels,
            "features": classes_features[c],
            "train_test_split_seed": seed
        }

    for c, model_dict in models_dict.items():
        num_epochs = len(model_dict["history"].history["loss"])
        print(
            f"Classification Report for {c} model after {num_epochs} epochs (Test Set):")
        print(model_dict["report"])

    model_name = get_basename(base_path)
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(models_dict, f)

    if args.plot:
        for c, model_dict in models_dict.items():
            plot_history(c,
                         model_dict["history"],
                         model_dict["confusion_matrix"],
                         classes)
        plt.tight_layout()
        plt.show()
