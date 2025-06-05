import argparse
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix


from keras.callbacks import EarlyStopping
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
    X, y = load_features(base_path)
    print("Loaded features")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    input_shape = X.shape[1:]
    model = get_model(input_shape, num_classes)
    print("Training model...")
    early_stopping_callback = EarlyStopping(
        patience=200, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        epochs=5000,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping_callback],
                        verbose=1)  # type: ignore

    print(
        f"Finished training model after {len(history.history['loss'])} epochs")  # type: ignore

    y_pred = model.predict(X_test, verbose=0)  # type: ignore
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    print("Classification Report (Test Set):")
    report = classification_report(
        y_test_labels, y_pred_labels, target_names=label_encoder.classes_, digits=4)
    print(report)

    if args.plot:
        plot_history(history,
                     y_test_labels, y_pred_labels,
                     label_encoder.classes_)

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    model_data = {
        "model_json": model.to_json(),
        "model_weights": model.get_weights(),
        "label_encoder": label_encoder,
        "classification_report": report,
        "confusion_matrix": cm,
        "history": history,
        "classes": label_encoder.classes_
    }

    model_name = get_basename(base_path)
    with open(f"{model_name}_model.pkl", "wb") as f:
        pickle.dump(model_data, f)
