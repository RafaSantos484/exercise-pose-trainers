import argparse
import pickle
from keras.models import Model
from keras.layers import Input, Conv1D, Concatenate, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
from keras.optimizers import Adam
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt


def get_model(angles_shape, points_shape, num_classes: int):
    # Input 1: Ângulos (shape: [n_angulos])
    input_angles = Input(shape=angles_shape, name='angles_input')
    x1 = Dense(64, activation='relu')(input_angles)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(32, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)

    # Input 2: Pontos 3D (shape: [n_pontos, 3])
    input_points = Input(shape=points_shape, name='points_input')
    x2 = Conv1D(128, kernel_size=4, activation='relu',
                padding='same')(input_points)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.3)(x2)
    x2 = GlobalAveragePooling1D()(x2)

    # Combinar os dois
    combined = Concatenate()([x1, x2])
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.3)(x)

    # Saída
    output = Dense(
        num_classes,
        activation='softmax' if num_classes > 2 else 'sigmoid'
    )(x)

    model = Model(inputs=[input_angles, input_points], outputs=output)
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

    classes_features = models_dict["classes_features"]
    classes = list(classes_features.keys())
    y_classes = models_dict["label_encoder"].classes_
    for c in classes:
        num_epochs = len(models_dict["models"][c]["history"].history["loss"])
        print(
            f"Classification Report for {c} model after {num_epochs} epochs (Test Set):")
        print(models_dict["models"][c]["report"])
        plot_history(c, models_dict["models"][c]["history"],
                     models_dict["models"][c]["confusion_matrix"], y_classes)
    plt.tight_layout()
    plt.show()
