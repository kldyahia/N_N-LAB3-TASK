"""Dataset loading, validation, encoding, and the task-prescribed balanced split."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FEATURES = (
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "OriginLocation",
    "BodyMass",
)
NUMERIC_FEATURES = ("CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass")
TARGET_COLUMN = "Species"


@dataclass(frozen=True)
class LoadedDataset:
    """A validated copy of the Penguins data with OriginLocation encoded numerically."""

    frame: pd.DataFrame
    origin_mapping: dict[str, float]
    species: tuple[str, ...]
    source_path: Path


def load_penguins(csv_path: str | Path) -> LoadedDataset:
    """Load the CSV without dropping rows; encode OriginLocation and retain numeric NaNs.

    Numeric missing values are intentionally not imputed here. They are imputed using
    medians fitted on the training split in ``preprocessing.py`` to avoid test leakage.
    """
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    frame = pd.read_csv(path)
    required = {TARGET_COLUMN, *FEATURES}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing_columns)}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Species contains missing values; class labels cannot be inferred safely.")

    original_row_count = len(frame)
    frame = frame.copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str)
    species = tuple(str(value) for value in pd.unique(frame[TARGET_COLUMN]))

    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # OriginLocation is nominal, but the assignment requests exactly two selectable
    # features and a 2-D decision-boundary plot. Encode it as one deterministic scalar.
    # This preserves the two-feature interface; it should not be interpreted as a
    # meaningful ordinal distance between locations.
    locations = frame["OriginLocation"].astype("string")
    known_locations = sorted(str(value) for value in locations.dropna().unique())
    if not known_locations:
        raise ValueError("OriginLocation has no usable categories.")
    mode_values = locations.dropna().mode()
    fallback_location = str(mode_values.iloc[0]) if not mode_values.empty else known_locations[0]
    locations = locations.fillna(fallback_location).astype(str)
    origin_mapping = {name: float(index) for index, name in enumerate(known_locations)}
    frame["OriginLocation"] = locations.map(origin_mapping).astype(float)

    if len(frame) != original_row_count:
        raise AssertionError("Loading/preprocessing unexpectedly changed the row count.")
    if set(FEATURES).difference(frame.columns):
        raise AssertionError("Internal feature validation failed.")

    return LoadedDataset(
        frame=frame,
        origin_mapping=origin_mapping,
        species=species,
        source_path=path,
    )


def create_balanced_split(
    dataset: LoadedDataset,
    feature_names: Iterable[str],
    class_a: str,
    class_b: str,
    train_per_class: int = 30,
    test_per_class: int = 20,
    random_state: int = 42,
):
    """Return raw train/test feature arrays and labels, with no repeated samples.

    Class A is encoded as -1 and Class B as +1. The source CSV is never modified and
    no rows are dropped during loading. Exactly 30 samples/class are used for training
    and the other 20/class are used for testing, as specified by Lab 3.
    """
    feature_names = tuple(feature_names)
    if len(feature_names) != 2:
        raise ValueError("Exactly two features must be selected.")
    if feature_names[0] == feature_names[1]:
        raise ValueError("The two selected features must be different.")
    invalid_features = set(feature_names).difference(FEATURES)
    if invalid_features:
        raise ValueError(f"Unsupported feature(s): {', '.join(sorted(invalid_features))}")
    if class_a == class_b:
        raise ValueError("Please select two different classes.")
    if class_a not in dataset.species or class_b not in dataset.species:
        raise ValueError("Selected class is not present in the dataset.")
    if train_per_class != 30 or test_per_class != 20:
        raise ValueError("The Lab 3 split requires 30 training and 20 testing samples per class.")

    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_name in (class_a, class_b):
        class_indices = dataset.frame.index[
            dataset.frame[TARGET_COLUMN] == class_name
        ].to_numpy()
        required_count = train_per_class + test_per_class
        if len(class_indices) != required_count:
            raise ValueError(
                f"Class '{class_name}' has {len(class_indices)} rows; expected exactly "
                f"{required_count} for the required 30/20 split."
            )
        shuffled = rng.permutation(class_indices)
        train_indices.extend(int(value) for value in shuffled[:train_per_class])
        test_indices.extend(int(value) for value in shuffled[train_per_class:required_count])

    train_frame = dataset.frame.loc[train_indices]
    test_frame = dataset.frame.loc[test_indices]
    x_train_raw = train_frame.loc[:, feature_names].to_numpy(dtype=float)
    x_test_raw = test_frame.loc[:, feature_names].to_numpy(dtype=float)
    y_train = np.where(train_frame[TARGET_COLUMN].to_numpy() == class_b, 1.0, -1.0)
    y_test = np.where(test_frame[TARGET_COLUMN].to_numpy() == class_b, 1.0, -1.0)

    if len(set(train_indices).intersection(test_indices)) != 0:
        raise AssertionError("Training and testing sets overlap.")
    if len(x_train_raw) != 60 or len(x_test_raw) != 40:
        raise AssertionError("Unexpected sample counts after the balanced split.")

    return {
        "train_frame": train_frame,
        "test_frame": test_frame,
        "x_train_raw": x_train_raw,
        "x_test_raw": x_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "train_indices": tuple(train_indices),
        "test_indices": tuple(test_indices),
    }
