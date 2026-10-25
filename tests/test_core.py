from pathlib import Path
import unittest

import numpy as np

from data_loader import FEATURES, load_penguins, create_balanced_split
from evaluation import accuracy_manual, confusion_matrix_manual
from experiment import classify_sample, run_experiment

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "penguins.csv"


class PenguinsTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_penguins(DATA_PATH)

    def test_loader_preserves_all_rows_and_encodes_origin(self):
        self.assertEqual(len(self.dataset.frame), 150)
        self.assertEqual(len(self.dataset.species), 3)
        self.assertEqual(set(self.dataset.origin_mapping), {"Biscoe", "Dream", "Torgersen"})
        self.assertTrue(np.issubdtype(self.dataset.frame["OriginLocation"].dtype, np.number))
        self.assertEqual(self.dataset.frame["Species"].isna().sum(), 0)

    def test_required_train_test_split_sizes_and_no_overlap(self):
        split = create_balanced_split(
            self.dataset,
            (FEATURES[0], FEATURES[1]),
            "Adelie",
            "Gentoo",
            random_state=42,
        )
        self.assertEqual(len(split["x_train_raw"]), 60)
        self.assertEqual(len(split["x_test_raw"]), 40)
        self.assertEqual(len(set(split["train_indices"]).intersection(split["test_indices"])), 0)
        self.assertEqual(np.count_nonzero(split["y_train"] == -1), 30)
        self.assertEqual(np.count_nonzero(split["y_train"] == 1), 30)
        self.assertEqual(np.count_nonzero(split["y_test"] == -1), 20)
        self.assertEqual(np.count_nonzero(split["y_test"] == 1), 20)

    def test_confusion_matrix_and_accuracy_are_manual(self):
        y_true = np.array([-1, -1, 1, 1])
        y_pred = np.array([-1, 1, 1, 1])
        np.testing.assert_array_equal(confusion_matrix_manual(y_true, y_pred), np.array([[1, 1], [0, 2]]))
        self.assertAlmostEqual(accuracy_manual(y_true, y_pred), 0.75)

    def test_both_models_complete_an_experiment(self):
        for algorithm in ("Perceptron", "Adaline"):
            result = run_experiment(
                self.dataset,
                ("CulmenLength", "BodyMass"),
                "Adelie",
                "Gentoo",
                algorithm=algorithm,
                learning_rate=0.01,
                epochs=30,
                mse_threshold=0.01,
                use_bias=True,
                random_state=42,
            )
            self.assertEqual(result.confusion_matrix.shape, (2, 2))
            self.assertEqual(len(result.predictions), 40)
            self.assertGreaterEqual(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)
            self.assertTrue(np.isfinite(result.x_train).all())
            self.assertTrue(np.isfinite(result.x_test).all())

    def test_single_sample_prediction_returns_one_selected_class(self):
        result = run_experiment(
            self.dataset,
            ("CulmenLength", "BodyMass"),
            "Adelie",
            "Gentoo",
            algorithm="Perceptron",
            learning_rate=0.01,
            epochs=50,
            use_bias=True,
            random_state=42,
        )
        sample = classify_sample(result, (39.1, 3750.0))
        self.assertIn(sample.predicted_class, {"Adelie", "Gentoo"})
        self.assertTrue(np.isfinite(sample.decision_score))
        self.assertEqual(len(sample.input_values), 2)

    def test_single_sample_rejects_wrong_number_of_features(self):
        result = run_experiment(
            self.dataset,
            ("CulmenLength", "BodyMass"),
            "Adelie",
            "Gentoo",
            algorithm="Perceptron",
            epochs=10,
            random_state=42,
        )
        with self.assertRaises(ValueError):
            classify_sample(result, (39.1,))


if __name__ == "__main__":
    unittest.main()
