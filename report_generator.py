"""Generate the Lab 3 analysis report, result table, and ten representative plots."""
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

from data_loader import FEATURES, LoadedDataset, load_penguins
from experiment import ExperimentResult, run_experiment
from visualization import create_decision_boundary_figure


@dataclass
class Candidate:
    result: ExperimentResult
    rank: int = -1


def _run_all_combinations(dataset: LoadedDataset, algorithm: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    class_pairs = list(itertools.combinations(dataset.species, 2))
    feature_pairs = list(itertools.combinations(FEATURES, 2))
    for features in feature_pairs:
        for class_a, class_b in class_pairs:
            result = run_experiment(
                dataset=dataset,
                feature_names=(features[0], features[1]),
                class_a=class_a,
                class_b=class_b,
                algorithm=algorithm,
                learning_rate=0.01,
                epochs=150,
                mse_threshold=0.01,
                use_bias=True,
                random_state=42,
            )
            candidates.append(Candidate(result=result))
    # Sorting is stable and makes report selection deterministic even in a tie.
    ordered = sorted(candidates, key=lambda item: item.result.accuracy)
    for rank, candidate in enumerate(ordered):
        candidate.rank = rank
    return ordered


def _select_representatives(candidates: list[Candidate]) -> list[tuple[str, Candidate]]:
    """Pick five ranked examples while ensuring five distinct feature pairs."""
    count = len(candidates)
    target_specs = [
        (count - 1, "Highest-performing example"),
        (round(0.75 * (count - 1)), "Upper-range example"),
        (round(0.50 * (count - 1)), "Middle-range example"),
        (round(0.25 * (count - 1)), "Lower-range example"),
        (0, "Lowest-performing example"),
    ]
    selected: list[tuple[str, Candidate]] = []
    used_feature_pairs: set[frozenset[str]] = set()
    used_class_pairs: set[frozenset[str]] = set()

    for target_rank, role in target_specs:
        eligible = [
            c for c in candidates
            if frozenset(c.result.feature_names) not in used_feature_pairs
        ]
        if not eligible:
            break
        chosen = min(
            eligible,
            key=lambda c: (
                abs(c.rank - target_rank)
                + (0.20 if frozenset((c.result.class_a, c.result.class_b)) in used_class_pairs else 0.0),
                c.rank,
            ),
        )
        selected.append((role, chosen))
        used_feature_pairs.add(frozenset(chosen.result.feature_names))
        used_class_pairs.add(frozenset((chosen.result.class_a, chosen.result.class_b)))
    return selected


def _write_all_results_csv(candidates_by_algorithm: dict[str, list[Candidate]], path: Path) -> None:
    fields = [
        "Algorithm", "Feature 1", "Feature 2", "Class A", "Class B",
        "Train samples", "Test samples", "Accuracy", "Class A recall", "Class B recall",
        "Confusion AA", "Confusion AB", "Confusion BA", "Confusion BB",
        "Epochs used", "Final MSE", "Learning rate", "MSE threshold", "Bias enabled",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for algorithm, candidates in candidates_by_algorithm.items():
            for candidate in candidates:
                result = candidate.result
                matrix = result.confusion_matrix
                writer.writerow({
                    "Algorithm": algorithm,
                    "Feature 1": result.feature_names[0],
                    "Feature 2": result.feature_names[1],
                    "Class A": result.class_a,
                    "Class B": result.class_b,
                    "Train samples": len(result.y_train),
                    "Test samples": len(result.y_test),
                    "Accuracy": f"{result.accuracy:.6f}",
                    "Class A recall": f"{result.metrics['class_a_recall']:.6f}",
                    "Class B recall": f"{result.metrics['class_b_recall']:.6f}",
                    "Confusion AA": int(matrix[0, 0]),
                    "Confusion AB": int(matrix[0, 1]),
                    "Confusion BA": int(matrix[1, 0]),
                    "Confusion BB": int(matrix[1, 1]),
                    "Epochs used": result.model.n_epochs_,
                    "Final MSE": f"{result.final_mse:.6f}",
                    "Learning rate": result.learning_rate,
                    "MSE threshold": result.mse_threshold,
                    "Bias enabled": result.use_bias,
                })


def _matrix_text(matrix) -> str:
    return (
        f"| Actual \\ Predicted | Class A | Class B |\n"
        f"|---|---:|---:|\n"
        f"| Class A | {int(matrix[0, 0])} | {int(matrix[0, 1])} |\n"
        f"| Class B | {int(matrix[1, 0])} | {int(matrix[1, 1])} |"
    )


def _qualitative_interpretation(result: ExperimentResult) -> str:
    accuracy = result.accuracy
    if accuracy >= 0.90:
        level = "strong separation on this test split"
    elif accuracy >= 0.75:
        level = "moderate-to-good separation on this test split"
    elif accuracy >= 0.60:
        level = "partial separation; some overlap or errors remain"
    else:
        level = "weak separation for this linear model on this test split"
    recall_a = result.metrics["class_a_recall"]
    recall_b = result.metrics["class_b_recall"]
    if abs(recall_a - recall_b) >= 0.25:
        balance = (
            f"Recall is uneven (Class A: {recall_a:.1%}, Class B: {recall_b:.1%}), "
            "so the overall accuracy should be read alongside the confusion matrix."
        )
    else:
        balance = f"Class recalls are Class A {recall_a:.1%} and Class B {recall_b:.1%}."
    return f"The model shows {level}. {balance} A linear classifier can only learn a straight decision boundary in the selected two-feature space."


def generate_analysis_report(
    dataset_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Run 30 combinations per algorithm and write a report plus ten plots."""
    dataset = load_penguins(dataset_path)
    out = Path(output_dir).expanduser().resolve()
    figures_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Remove old generated plots so repeated report runs do not leave stale figures.
    for old_figure in figures_dir.glob("*.png"):
        old_figure.unlink()

    results_by_algorithm = {
        "Perceptron": _run_all_combinations(dataset, "Perceptron"),
        "Adaline": _run_all_combinations(dataset, "Adaline"),
    }
    _write_all_results_csv(results_by_algorithm, out / "experiment_results.csv")

    report: list[str] = [
        "# Penguins Binary Classification — Lab 3 Analysis Report",
        "",
        "## 1. Objective",
        "Implement and compare Perceptron and Adaline on two-class Penguins classification tasks using exactly two selected features, as required by Lab 3.",
        "",
        "## 2. Dataset and preprocessing",
        f"- Dataset: `{dataset.source_path.name}`.",
        f"- Rows preserved during loading: **{len(dataset.frame)}**.",
        f"- Classes: {', '.join(dataset.species)} (the selected binary experiment uses the two classes shown in each result).",
        f"- Selectable features: {', '.join(FEATURES)}.",
        "- Numeric missing values are imputed with medians calculated from the training split only; no row is dropped.",
        "- `OriginLocation` is mapped to a deterministic numeric code so it remains one selectable feature and the visualization stays two-dimensional. This coding is nominal, not ordinal; numeric distance between location codes has no real-world meaning.",
        "- Each class contributes 30 non-repeated training samples and the remaining 20 testing samples. The fixed random seed is 42 so runs are reproducible.",
        "- The two selected features are standardized with training-set means and standard deviations. Test data use the same training-derived values.",
        "- Class A is encoded as -1; Class B as +1. Both decision boundaries are linear.",
        "- No scikit-learn package or ready-made confusion-matrix utility is used. Confusion matrices and overall accuracy are computed by custom loops in `evaluation.py`.",
        "",
        "## 3. Training settings",
        "The generated comparison uses learning rate = 0.01, maximum epochs = 150, Bias enabled, and MSE threshold = 0.01 for Adaline. The GUI lets the user change these values for an individual run.",
        "Perceptron updates weights when a sample is misclassified. Adaline uses stochastic gradient descent and updates its weights using the current sample's linear-output error; MSE is checked after each epoch.",
        "",
        "## 4. Representative plots and analysis",
        "Five different feature pairs are shown for each algorithm. Examples are selected across the observed performance ranking, from high to low. All 30 feature-pair/class-pair results per algorithm are included in `experiment_results.csv`.",
        "",
    ]

    for algorithm, all_candidates in results_by_algorithm.items():
        selected = _select_representatives(all_candidates)
        report.extend([f"### 4.{1 if algorithm == 'Perceptron' else 2}. {algorithm}", ""])
        for index, (role, candidate) in enumerate(selected, start=1):
            result = candidate.result
            safe_name = "_".join(result.feature_names).lower()
            figure_name = (
                f"{algorithm.lower()}_{index:02d}_{safe_name}_"
                f"{result.class_a.lower()}_vs_{result.class_b.lower()}.png"
            )
            figure_path = figures_dir / figure_name
            fig = create_decision_boundary_figure(result, dataset.origin_mapping)
            fig.savefig(figure_path, bbox_inches="tight", dpi=150)

            report.extend([
                f"#### {index}. {role}",
                f"- **Classes:** {result.class_a} (Class A, -1) vs {result.class_b} (Class B, +1).",
                f"- **Features:** `{result.feature_names[0]}` and `{result.feature_names[1]}`.",
                f"- **Test accuracy:** {result.accuracy:.1%} ({int(result.confusion_matrix[0, 0] + result.confusion_matrix[1, 1])}/40 correct).",
                f"- **Epochs used:** {result.model.n_epochs_}; **final training MSE:** {result.final_mse:.4f}.",
                "",
                _matrix_text(result.confusion_matrix),
                "",
                f"![{algorithm} decision boundary — {result.feature_names[0]} and {result.feature_names[1]}]({('figures/' + figure_name)})",
                "",
                f"**Interpretation:** {_qualitative_interpretation(result)}",
                "",
            ])

        best_accuracy = max(candidate.result.accuracy for candidate in all_candidates)
        best = [candidate.result for candidate in all_candidates if abs(candidate.result.accuracy - best_accuracy) < 1e-12]
        best_descriptions = "; ".join(
            f"`{item.feature_names[0]} + {item.feature_names[1]}` ({item.class_a} vs {item.class_b})"
            for item in best
        )
        report.extend([
            f"**Highest test accuracy for {algorithm}: {best_accuracy:.1%}.**",
            f"Best-performing feature/class combination(s): {best_descriptions}.",
            "",
        ])

    report.extend([
        "## 5. Overall comparison",
        "Use the full `experiment_results.csv` to compare all 30 combinations per algorithm. A high test score means the selected linear boundary separated the held-out samples well in that split; a lower score may indicate feature overlap, a non-linear class relationship, or model settings that did not converge sufficiently. These observations apply to this fixed split and should not be treated as a universal performance guarantee.",
        "",
        "## 6. Important implementation notes",
        "- The plotted decision boundary is derived from the trained weights and bias after applying the same training-fitted preprocessing to the plotted grid.",
        "- Circle markers represent training samples; triangle markers represent test samples. Test markers are hollow to make the held-out points easy to identify.",
        "- `OriginLocation` encoding is a practical compromise to satisfy the two-feature/2-D plot requirement. For a production model, a categorical method designed for nominal variables would be preferable, but one-hot encoding would expand this selected feature into multiple input dimensions.",
        "- Each run uses a fixed train/test split seed for fair comparison. Changing the seed is a useful follow-up sensitivity check, but is not part of this fixed-seed report.",
        "",
        "## 7. Deliverables",
        "- `experiment_results.csv`: all tested feature-pair/class-pair combinations (30 per algorithm).",
        "- `figures/`: five representative plots per algorithm.",
        "- This Markdown report with confusion matrices and interpretations.",
        "",
    ])

    report_path = out / "analysis_report.md"
    markdown_text = "\n".join(report)
    report_path.write_text(markdown_text, encoding="utf-8")

    # Also create a browser-friendly HTML report while keeping Markdown as the editable source.
    import mistune

    markdown_to_html = mistune.create_markdown(plugins=["table"])
    html_body = markdown_to_html(markdown_text)
    html_page = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Penguins Binary Classification — Lab 3 Report</title>
<style>
:root {{ color-scheme: light; }}
body {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 60px; font-family: Segoe UI, Arial, sans-serif; line-height: 1.6; color: #182230; background: #f8fafc; }}
main {{ background: white; padding: 36px; border: 1px solid #e2e8f0; border-radius: 14px; box-shadow: 0 8px 28px rgba(15, 23, 42, .06); }}
h1, h2, h3, h4 {{ line-height: 1.25; color: #0f172a; }}
h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 14px; }}
h2 {{ margin-top: 34px; padding-bottom: 8px; border-bottom: 1px solid #e2e8f0; }}
h3 {{ margin-top: 28px; }}
img {{ display: block; max-width: 100%; height: auto; margin: 18px auto; border: 1px solid #e2e8f0; border-radius: 8px; }}
table {{ border-collapse: collapse; width: auto; max-width: 100%; margin: 14px 0 22px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
th {{ background: #eff6ff; }}
code {{ background: #f1f5f9; border-radius: 4px; padding: 2px 4px; }}
blockquote {{ border-left: 4px solid #93c5fd; margin-left: 0; padding-left: 16px; color: #475569; }}
@media (max-width: 700px) {{ body {{ padding: 12px; }} main {{ padding: 18px; }} }}
</style>
</head>
<body><main>
{html_body}
</main></body></html>
"""
    (out / "analysis_report.html").write_text(html_page, encoding="utf-8")
    return report_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the Lab 3 Penguins experiment report.")
    default_dataset = Path(__file__).resolve().parent / "data" / "penguins.csv"
    parser.add_argument("--dataset", default=str(default_dataset), help="Path to penguins.csv")
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parent / "output"),
        help="Output directory for the report, CSV, and figures",
    )
    args = parser.parse_args()
    report_path = generate_analysis_report(args.dataset, args.output)
    print(f"Report created: {report_path}")
    print(f"HTML report: {Path(args.output).resolve() / 'analysis_report.html'}")
    print(f"Results table: {Path(args.output).resolve() / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
