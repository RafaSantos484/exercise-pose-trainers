import argparse
import pickle
from keras.models import Sequential
from keras.layers import Conv1D, Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt


def get_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),

        Conv1D(256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),

        GlobalAveragePooling1D(),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['categorical_accuracy']
    )

    return model


def plot_history(history, y_true, y_pred, classes):
    # Accuracy
    plt.figure()
    plt.plot(history.history['categorical_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_categorical_accuracy'],
             label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(colorbar=False)
    disp.ax_.set_title("CNN Confusion Matrix")

    plt.tight_layout()
    plt.show()


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
    disp.ax_.set_title("CNN Confusion Matrix")

    plt.tight_layout()
    plt.show()
