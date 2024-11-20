import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from features import (
    LONG_TERM_STRENGTH_FEATURES_GENERAL,
    YIELD_STRENGTH_FEATURES_GENERAL,
    YIELD_STRENGTH_FEATURES_THERMOCALC,
    TOUGHNESS_FEATURES,
)


def parse_args():
    """Parse cmd arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="long-term-strength",
        help="Target property ('yield-strength' or 'long-term-strength', or 'toughness')",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="general",
        help="Model type ('general' or 'specific')",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/dataset/ys_subset.csv",
        help="Path to data in csv format",
    )
    args = parser.parse_args()
    return args


def load_long_term_strength_model() -> CatBoostRegressor:
    """Load long-term strength model"""

    model: CatBoostRegressor = joblib.load(
        "./data/models/long_term_strength/cb_long_strength_model.joblib"
    )
    return model

def load_toughness_model() -> CatBoostRegressor:
    """Load toughness model"""

    model: CatBoostRegressor = joblib.load(
        "./data/models/toughness/general/toughness_general.joblib"
    )
    return model


def load_data(data_path: str) -> pd.DataFrame:
    """Load long term strength data
    Args:
        data_path: path to data in csv format

    """
    data = pd.read_csv(data_path)
    return data


def load_yield_strength_model_general() -> CatBoostRegressor:
    """Load long-term strength model"""

    model: CatBoostRegressor = joblib.load(
        "./data/models/yield_strength/general/ys_general_model.joblib"
    )
    return model


def infer_thremo_calc_models(
    models_path_list: list[Path], np_data: np.ndarray
) -> np.ndarray:
    """Infer thermo calc models to calculate thermo calc features
    Args:
        models_path_list: list of models paths
    Returns:
        predicted thermo calc features

    """
    models = [joblib.load(p) for p in models_path_list]
    predictions_list = []
    for model in models:
        tc_prediction = model.predict(np_data)
        predictions_list.append(tc_prediction)
    predictions = np.asarray(predictions_list).T
    return predictions


def main(args) -> None:
    """Main function to infer models."""

    target: str = args.target
    model_type: str = args.model_type
    data_path: str = args.data_path

    data: pd.DataFrame = load_data(data_path)

    np_data: np.ndarray
    if target == "long-term-strength":
        assert (
            model_type == "general"
        ), "For long term strength only 'general' model is available"
        model = load_long_term_strength_model()
        data["logt"] = np.log10(data["long-therm strength time"])
        np_data = np.asarray(data[LONG_TERM_STRENGTH_FEATURES_GENERAL])
        predictions = model.predict(np_data)

    elif target == "yield-strength":
        if model_type == "general":
            model: CatBoostRegressor = load_yield_strength_model_general()
            np_data = np.asarray(data[YIELD_STRENGTH_FEATURES_GENERAL])
            predictions = model.predict(np_data)
        elif model_type == "specific":
            thermo_calc_models_paths: list[Path] = list(
                Path("./data/models/yield_strength/specific").glob("*.joblib")
            )
            head_model = thermo_calc_models_paths.pop(0)
            np_data = np.asarray(data[YIELD_STRENGTH_FEATURES_THERMOCALC])
            thermo_calc_features_predicted: np.ndarray = infer_thremo_calc_models(
                models_path_list=thermo_calc_models_paths, np_data=np_data
            )
            concatenated_data = np.hstack(
                (
                    np_data,
                    thermo_calc_features_predicted,
                    np.asarray(data["Temperature"]).reshape(len(data), 1),
                )
            )
            kernel_ridge_model: Pipeline = joblib.load(head_model)
            predictions = kernel_ridge_model.predict(concatenated_data)
        else:
            raise ValueError(
                f"Unknown model_type. Expected to be 'general' or 'specific', got: {model_type}"
            )
    elif target == "toughness":
        assert (
            model_type == "general"
        ), "For toghness only 'general' model is available"
        model = load_toughness_model()
        np_data = np.asarray(data[TOUGHNESS_FEATURES])
        predictions = model.predict(np_data)
    else:
        raise ValueError(
            f"Unknown target. Expected to be 'yield-strength', or 'long-term-strength', got: {target}"
        )

    data[f"{target} predicted by {model_type}"] = predictions

    save_path = f"./data/results/{target}_{model_type}.csv"
    Path("./data/results").mkdir(exist_ok=True, parents=True)
    data.to_csv(save_path)
    print(f"{target} predicted by {model_type} saved to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
