import argparse
import pickle
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str,
                        help="Path to the model folder")
    args = parser.parse_args()

    with open(args.model_path, "rb") as f:
        models_dict: dict = pickle.load(f)

    for c, model_dict in models_dict.items():
        model = model_dict["model"]
        shape = [None, len(model_dict["features"]["angles"])]
        initial_type = [("input", FloatTensorType(shape))]
        onx = convert_sklearn(
            model, initial_types=initial_type, options={"zipmap": False})
        with open(f"{c}_model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
