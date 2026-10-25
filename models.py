"""Implementations of the Lab 3 Perceptron and Adaline algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LinearBinaryClassifier(ABC):
    """Shared linear decision function for binary labels encoded as -1 and +1."""

    algorithm_name = "Linear Binary Classifier"

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        use_bias: bool = True,
        random_state: int = 42,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        if epochs < 1:
            raise ValueError("Epochs must be at least 1.")
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.use_bias = bool(use_bias)
        self.random_state = int(random_state)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.history: list[dict[str, float | int]] = []
        self.n_epochs_: int = 0
        self.fitted_: bool = False

    def _initialize(self, feature_count: int) -> np.random.Generator:
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.normal(loc=0.0, scale=0.01, size=feature_count)
        self.bias = float(rng.normal(0.0, 0.01)) if self.use_bias else 0.0
        self.history = []
        self.fitted_ = False
        return rng

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.weights + self.bias

    def predict(self, x: np.ndarray) -> np.ndarray:
        # At net input zero, choose class +1 consistently.
        return np.where(self.decision_function(x) >= 0.0, 1.0, -1.0)

    def _record_epoch(self, epoch: int, x: np.ndarray, y: np.ndarray, errors: int | None = None) -> None:
        scores = x @ self.weights + self.bias
        mse = float(np.mean((y - scores) ** 2))
        record: dict[str, float | int] = {"epoch": int(epoch), "mse": mse}
        if errors is not None:
            record["errors"] = int(errors)
        self.history.append(record)

    def _check_training_data(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("The Lab 3 classifier expects exactly two input features.")
        if len(x) != len(y) or len(y) == 0:
            raise ValueError("Feature and target sample counts must match and be non-zero.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training data must be imputed and finite before fitting.")
        if not set(np.unique(y)).issubset({-1.0, 1.0}):
            raise ValueError("Targets must be encoded as -1 and +1.")
        return x, y

    def _check_fitted(self) -> None:
        if not self.fitted_ or self.weights is None:
            raise RuntimeError("Model has not been trained yet.")

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearBinaryClassifier":
        raise NotImplementedError


class PerceptronClassifier(LinearBinaryClassifier):
    """Sample-by-sample Perceptron updates using the classification error."""

    algorithm_name = "Perceptron"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PerceptronClassifier":
        x, y = self._check_training_data(x, y)
        rng = self._initialize(x.shape[1])

        for epoch in range(1, self.epochs + 1):
            mistakes = 0
            for index in rng.permutation(len(x)):
                score = float(x[index] @ self.weights + self.bias)
                prediction = 1.0 if score >= 0.0 else -1.0
                if prediction != y[index]:
                    update = self.learning_rate * (y[index] - prediction)
                    self.weights += update * x[index]
                    if self.use_bias:
                        self.bias += float(update)
                    mistakes += 1
            self._record_epoch(epoch, x, y, mistakes)
            if mistakes == 0:
                break

        self.n_epochs_ = len(self.history)
        self.fitted_ = True
        return self


class AdalineClassifier(LinearBinaryClassifier):
    """ADALINE with a linear output and stochastic gradient descent on MSE."""

    algorithm_name = "Adaline"

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 100,
        use_bias: bool = True,
        mse_threshold: float = 0.01,
        random_state: int = 42,
    ) -> None:
        super().__init__(learning_rate, epochs, use_bias, random_state)
        if mse_threshold < 0:
            raise ValueError("MSE threshold cannot be negative.")
        self.mse_threshold = float(mse_threshold)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "AdalineClassifier":
        x, y = self._check_training_data(x, y)
        rng = self._initialize(x.shape[1])

        for epoch in range(1, self.epochs + 1):
            # Update the weights after each sample using the linear output error.
            for index in rng.permutation(len(x)):
                output = float(x[index] @ self.weights + self.bias)
                error = y[index] - output
                self.weights += self.learning_rate * error * x[index]
                if self.use_bias:
                    self.bias += self.learning_rate * float(error)

            self._record_epoch(epoch, x, y)
            current_mse = float(self.history[-1]["mse"])
            if not np.isfinite(current_mse) or not np.isfinite(self.weights).all():
                raise ValueError(
                    f"Adaline diverged at epoch {epoch}. Try a smaller learning rate."
                )
            if current_mse <= self.mse_threshold:
                break

        self.n_epochs_ = len(self.history)
        self.fitted_ = True
        return self
