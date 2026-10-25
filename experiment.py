"""Application service: split data, preprocess, train, and evaluate an experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from data_loader import LoadedDataset, create_balanced_split
from evaluation import accuracy_manual, confusion_matrix_manual, metrics_from_confusion
from models import AdalineClassifier, LinearBinaryClassifier, PerceptronClassifier
from preprocessing import Standardizer

AlgorithmName = Literal["Perceptron", "Adaline"]


@dataclass
class ExperimentResult:
    algorithm: AlgorithmName
    feature_names: tuple[str, str]
    class_a: str
    class_b: str
    learning_rate: float
    epochs_requested: int
    mse_threshold: float
    use_bias: bool
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    scaler: Standardizer
    model: LinearBinaryClassifier
    confusion_matrix: np.ndarray
    accuracy: float
    metrics: dict[str, float]

    @property
    def final_mse(self) -> float:
        if not self.model.history:
            return float("nan")
        return float(self.model.history[-1]["mse"])


@dataclass(frozen=True)
class SamplePrediction:
    """Prediction details for one new sample entered through the GUI."""

    predicted_class: str
    decision_score: float
    input_values: tuple[float, float]


def classify_sample(result: ExperimentResult, feature_values: tuple[float, float] | list[float]) -> SamplePrediction:
    """Classify one new sample using the preprocessing fitted during training."""
    values = np.asarray(feature_values, dtype=float).reshape(-1)
    if values.shape != (2,):
        raise ValueError("Enter exactly two feature values for the selected model.")
    if not np.isfinite(values).all():
        raise ValueError("Sample values must be valid finite numbers.")

    # Keep the same imputation and scaling parameters that were fitted on training data.
    scaled_values = result.scaler.transform(values.reshape(1, 2))
    score = float(result.model.decision_function(scaled_values)[0])
    predicted_code = float(result.model.predict(scaled_values)[0])
    predicted_class = result.class_b if predicted_code == 1.0 else result.class_a
    return SamplePrediction(
        predicted_class=predicted_class,
        decision_score=score,
        input_values=(float(values[0]), float(values[1])),
    )


def run_experiment(
    dataset: LoadedDataset,
    feature_names: tuple[str, str],
    class_a: str,
    class_b: str,
    algorithm: AlgorithmName = "Perceptron",
    learning_rate: float = 0.01,
    epochs: int = 100,
    mse_threshold: float = 0.01,
    use_bias: bool = True,
    random_state: int = 42,
) -> ExperimentResult:
    """Run one repeatable, balanced experiment using exactly two features."""
    split = create_balanced_split(
        dataset,
        feature_names,
        class_a,
        class_b,
        train_per_class=30,
        test_per_class=20,
        random_state=random_state,
    )

    scaler = Standardizer()
    x_train = scaler.fit_transform(split["x_train_raw"])
    x_test = scaler.transform(split["x_test_raw"])

    if algorithm == "Perceptron":
        model: LinearBinaryClassifier = PerceptronClassifier(
            learning_rate=learning_rate,
            epochs=epochs,
            use_bias=use_bias,
            random_state=random_state,
        )
    elif algorithm == "Adaline":
        model = AdalineClassifier(
            learning_rate=learning_rate,
            epochs=epochs,
            use_bias=use_bias,
            mse_threshold=mse_threshold,
            random_state=random_state,
        )
    else:
        raise ValueError("Algorithm must be 'Perceptron' or 'Adaline'.")

    model.fit(x_train, split["y_train"])
    predictions = model.predict(x_test)
    matrix = confusion_matrix_manual(split["y_test"], predictions)
    accuracy = accuracy_manual(split["y_test"], predictions)

    return ExperimentResult(
        algorithm=algorithm,
        feature_names=tuple(feature_names),
        class_a=class_a,
        class_b=class_b,
        learning_rate=float(learning_rate),
        epochs_requested=int(epochs),
        mse_threshold=float(mse_threshold),
        use_bias=bool(use_bias),
        train_indices=split["train_indices"],
        test_indices=split["test_indices"],
        x_train_raw=split["x_train_raw"],
        x_test_raw=split["x_test_raw"],
        x_train=x_train,
        x_test=x_test,
        y_train=split["y_train"],
        y_test=split["y_test"],
        predictions=predictions,
        scaler=scaler,
        model=model,
        confusion_matrix=matrix,
        accuracy=accuracy,
        metrics=metrics_from_confusion(matrix),
    )
