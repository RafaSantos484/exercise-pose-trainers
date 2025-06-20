import argparse
import pickle
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def get_model(input_shape, num_classes: int):
    model = Sequential([
        Input(shape=input_shape),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['categorical_accuracy']
    )

    return model


def plot_history(model_name: str, history, confusion_matrix, classes):
    # Accuracy
    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'],
             label='Validation Accuracy')
    plt.title(f'{model_name} Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

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

    for c, model_dict in models_dict.items():
        print(model_dict["classes"])
        num_epochs = len(model_dict["history"].history["loss"])
        print(
            f"Classification Report for {c} model after {num_epochs} epochs (Test Set):")
        print(model_dict["report"])
        plot_history(c,
                     model_dict["history"], model_dict["confusion_matrix"],
                     model_dict["classes"])
    plt.tight_layout()
    plt.show()
