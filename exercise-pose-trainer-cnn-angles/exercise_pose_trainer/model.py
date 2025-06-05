import argparse
import pickle
from keras.models import Sequential
from keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout, SpatialDropout1D, BatchNormalization, ReLU
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt


def get_model(input_shape, num_classes):
    model = Sequential()

    # Bloco 1
    model.add(Conv1D(64, 3, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(SpatialDropout1D(0.3))

    # Bloco 2
    model.add(Conv1D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(SpatialDropout1D(0.3))

    # Bloco 3
    model.add(Conv1D(256, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(SpatialDropout1D(0.3))

    # Bloco 4
    model.add(Conv1D(512, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(SpatialDropout1D(0.3))

    # Bottleneck com convolução 1x1
    model.add(Conv1D(128, 1, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())

    # Bloco extra 5 (com residual simplificado)
    model.add(Conv1D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(SpatialDropout1D(0.3))

    # Pooling global
    model.add(GlobalAveragePooling1D())

    # Camadas densas mais robustas
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))

    # Saída
    model.add(Dense(num_classes, activation="softmax"))

    # Compilação
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
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
