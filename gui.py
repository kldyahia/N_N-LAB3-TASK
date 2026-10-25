"""Tkinter GUI only. Training, preprocessing, evaluation, and plotting logic live elsewhere."""
from __future__ import annotations

from pathlib import Path
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from data_loader import FEATURES, LoadedDataset, load_penguins
from experiment import ExperimentResult, classify_sample, run_experiment
from report_generator import generate_analysis_report
from visualization import create_decision_boundary_figure


class PenguinsClassifierGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Penguins Binary Classifier — Neural Networks Lab 3")
        self.root.geometry("1380x870")
        self.root.minsize(1120, 720)

        project_dir = Path(__file__).resolve().parent
        default_dataset = project_dir / "data" / "penguins.csv"
        self.dataset_path_var = tk.StringVar(value=str(default_dataset))
        self.dataset: LoadedDataset = load_penguins(default_dataset)
        self.last_result: ExperimentResult | None = None
        self.last_signature: tuple | None = None
        self.canvas: FigureCanvasTkAgg | None = None
        self.figure = None
        self.sample_fields: list[tuple[str, tk.StringVar, str]] = []

        self.feature_1_var = tk.StringVar(value=FEATURES[0])
        self.feature_2_var = tk.StringVar(value=FEATURES[1])
        self.class_a_var = tk.StringVar(value=self.dataset.species[0])
        self.class_b_var = tk.StringVar(value=self.dataset.species[-1])
        self.algorithm_var = tk.StringVar(value="Perceptron")
        self.learning_rate_var = tk.StringVar(value="0.01")
        self.epochs_var = tk.StringVar(value="100")
        self.mse_threshold_var = tk.StringVar(value="0.01")
        self.use_bias_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready. Choose two features and two classes, then train the model.")
        self.accuracy_var = tk.StringVar(value="—")
        self.model_info_var = tk.StringVar(value="No trained model yet.")
        self.confusion_var = tk.StringVar(value="Train a model to see the manually calculated confusion matrix.")
        self.prediction_var = tk.StringVar(value="Train a model before classifying a single sample.")
        self.report_status_var = tk.StringVar(value="The report runs all 30 feature-pair/class-pair combinations per algorithm.")

        self._configure_style()
        self._build_layout()
        self._refresh_sample_inputs()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 9), foreground="#475569")
        style.configure("Metric.TLabel", font=("Segoe UI", 19, "bold"), foreground="#1d4ed8")
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Penguins Binary Classifier", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Perceptron and Adaline • two features • balanced 30/20 split per class • manual confusion matrix",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self.train_tab = ttk.Frame(self.notebook, padding=10)
        self.predict_tab = ttk.Frame(self.notebook, padding=18)
        self.report_tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(self.train_tab, text="Train & Evaluate")
        self.notebook.add(self.predict_tab, text="Classify One Sample")
        self.notebook.add(self.report_tab, text="Full Analysis Report")

        self._build_train_tab()
        self._build_predict_tab()
        self._build_report_tab()

        footer = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        footer.pack(fill="x")
        ttk.Separator(footer).pack(fill="x", pady=(0, 6))
        ttk.Label(footer, textvariable=self.status_var, wraplength=1300).pack(anchor="w")

    def _build_train_tab(self) -> None:
        self.train_tab.columnconfigure(0, weight=0, minsize=320)
        self.train_tab.columnconfigure(1, weight=1)
        self.train_tab.rowconfigure(0, weight=1)

        config = ttk.LabelFrame(self.train_tab, text="Experiment setup", style="Section.TLabelframe", padding=12)
        config.grid(row=0, column=0, sticky="nsw", padx=(0, 10), pady=2)
        config.columnconfigure(0, weight=1)

        ttk.Label(config, text="Penguins CSV").grid(row=0, column=0, sticky="w")
        path_row = ttk.Frame(config)
        path_row.grid(row=1, column=0, sticky="ew", pady=(3, 10))
        path_row.columnconfigure(0, weight=1)
        ttk.Entry(path_row, textvariable=self.dataset_path_var, width=25).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_row, text="Browse…", command=self._browse_dataset).grid(row=0, column=1, padx=(5, 0))

        ttk.Label(config, text="Feature 1").grid(row=2, column=0, sticky="w")
        self.feature_1_box = ttk.Combobox(config, textvariable=self.feature_1_var, values=FEATURES, state="readonly")
        self.feature_1_box.grid(row=3, column=0, sticky="ew", pady=(3, 8))
        self.feature_1_box.bind("<<ComboboxSelected>>", self._on_feature_changed)

        ttk.Label(config, text="Feature 2").grid(row=4, column=0, sticky="w")
        self.feature_2_box = ttk.Combobox(config, textvariable=self.feature_2_var, values=FEATURES, state="readonly")
        self.feature_2_box.grid(row=5, column=0, sticky="ew", pady=(3, 8))
        self.feature_2_box.bind("<<ComboboxSelected>>", self._on_feature_changed)

        class_row = ttk.Frame(config)
        class_row.grid(row=6, column=0, sticky="ew")
        class_row.columnconfigure(0, weight=1)
        class_row.columnconfigure(1, weight=1)
        ttk.Label(class_row, text="Class A (-1)").grid(row=0, column=0, sticky="w")
        ttk.Label(class_row, text="Class B (+1)").grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.class_a_box = ttk.Combobox(class_row, textvariable=self.class_a_var, values=self.dataset.species, state="readonly", width=13)
        self.class_b_box = ttk.Combobox(class_row, textvariable=self.class_b_var, values=self.dataset.species, state="readonly", width=13)
        self.class_a_box.grid(row=1, column=0, sticky="ew", pady=(3, 10))
        self.class_b_box.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(3, 10))

        ttk.Label(config, text="Algorithm").grid(row=7, column=0, sticky="w")
        algorithm_frame = ttk.Frame(config)
        algorithm_frame.grid(row=8, column=0, sticky="w", pady=(3, 10))
        ttk.Radiobutton(algorithm_frame, text="Perceptron", variable=self.algorithm_var, value="Perceptron").pack(side="left")
        ttk.Radiobutton(algorithm_frame, text="Adaline", variable=self.algorithm_var, value="Adaline").pack(side="left", padx=(12, 0))

        params = ttk.Frame(config)
        params.grid(row=9, column=0, sticky="ew")
        params.columnconfigure(1, weight=1)
        ttk.Label(params, text="Learning rate").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(params, textvariable=self.learning_rate_var, width=12).grid(row=0, column=1, sticky="e", pady=3)
        ttk.Label(params, text="Epochs (max)").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(params, textvariable=self.epochs_var, width=12).grid(row=1, column=1, sticky="e", pady=3)
        ttk.Label(params, text="MSE threshold").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(params, textvariable=self.mse_threshold_var, width=12).grid(row=2, column=1, sticky="e", pady=3)
        ttk.Checkbutton(config, text="Use bias", variable=self.use_bias_var).grid(row=10, column=0, sticky="w", pady=(8, 10))

        ttk.Label(
            config,
            text="Training: 30 samples/class\nTesting: remaining 20 samples/class\nSplit seed: 42",
            justify="left",
        ).grid(row=11, column=0, sticky="w", pady=(0, 10))
        ttk.Button(config, text="Train & Evaluate", style="Action.TButton", command=self._train_model).grid(row=12, column=0, sticky="ew")

        result_panel = ttk.Frame(self.train_tab)
        result_panel.grid(row=0, column=1, sticky="nsew", pady=2)
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(1, weight=1)

        summary = ttk.LabelFrame(result_panel, text="Evaluation summary", style="Section.TLabelframe", padding=10)
        summary.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        summary.columnconfigure(0, weight=1)
        ttk.Label(summary, text="Overall test accuracy").grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.accuracy_var, style="Metric.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.model_info_var, wraplength=900, justify="left").grid(row=2, column=0, sticky="w", pady=(5, 0))
        ttk.Label(summary, textvariable=self.confusion_var, justify="left", font=("Consolas", 10)).grid(row=3, column=0, sticky="w", pady=(8, 0))

        self.plot_frame = ttk.LabelFrame(result_panel, text="Decision boundary", style="Section.TLabelframe", padding=5)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        ttk.Label(
            self.plot_frame,
            text="Your plot will appear here after training.",
        ).grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _build_predict_tab(self) -> None:
        self.predict_tab.columnconfigure(0, weight=1)
        intro = ttk.LabelFrame(self.predict_tab, text="Single-sample prediction", style="Section.TLabelframe", padding=16)
        intro.grid(row=0, column=0, sticky="nw")
        ttk.Label(
            intro,
            text="Train a model on the first tab, then enter values for the same two selected features.",
            wraplength=720,
        ).grid(row=0, column=0, sticky="w", pady=(0, 14))
        self.sample_fields_frame = ttk.Frame(intro)
        self.sample_fields_frame.grid(row=1, column=0, sticky="w")
        ttk.Button(intro, text="Classify sample", style="Action.TButton", command=self._predict_sample).grid(row=2, column=0, sticky="w", pady=(15, 10))
        ttk.Label(intro, textvariable=self.prediction_var, justify="left", wraplength=760, font=("Segoe UI", 11)).grid(row=3, column=0, sticky="w", pady=(5, 0))
        ttk.Label(
            self.predict_tab,
            text="Note: the decision score is not a probability. The prediction is based on the learned linear decision function.",
            wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(16, 0))

    def _build_report_tab(self) -> None:
        box = ttk.LabelFrame(self.report_tab, text="Automatic analysis report", style="Section.TLabelframe", padding=18)
        box.pack(anchor="nw", fill="x")
        ttk.Label(
            box,
            text=(
                "Generate the report required by Lab 3. It evaluates all 10 two-feature combinations "
                "against all 3 class pairs for both algorithms, writes the full results table, and saves "
                "five representative plots per algorithm with confusion matrices and written analysis."
            ),
            wraplength=900,
            justify="left",
        ).pack(anchor="w", pady=(0, 12))
        self.generate_report_button = ttk.Button(box, text="Generate full report", style="Action.TButton", command=self._generate_report)
        self.generate_report_button.pack(anchor="w")
        ttk.Label(box, textvariable=self.report_status_var, wraplength=1000, justify="left").pack(anchor="w", pady=(12, 0))
        self.open_report_button = ttk.Button(box, text="Open report in browser", command=self._open_report, state="disabled")
        self.open_report_button.pack(anchor="w", pady=(12, 0))
        ttk.Label(
            self.report_tab,
            text="Output files are written to the project's output/ folder. The Markdown report links to the generated PNG figures.",
            wraplength=900,
        ).pack(anchor="w", pady=(15, 0))

    def _browse_dataset(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select penguins.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            self.dataset_path_var.set(chosen)
            try:
                self._reload_dataset_if_needed(force=True)
                self.status_var.set(f"Loaded dataset: {self.dataset_path_var.get()}")
            except Exception as exc:
                messagebox.showerror("Dataset error", str(exc), parent=self.root)

    def _reload_dataset_if_needed(self, force: bool = False) -> None:
        selected_path = Path(self.dataset_path_var.get()).expanduser().resolve()
        if not force and selected_path == self.dataset.source_path:
            return
        bundle = load_penguins(selected_path)
        self.dataset = bundle
        self.class_a_box.configure(values=bundle.species)
        self.class_b_box.configure(values=bundle.species)
        if self.class_a_var.get() not in bundle.species:
            self.class_a_var.set(bundle.species[0])
        if self.class_b_var.get() not in bundle.species or self.class_b_var.get() == self.class_a_var.get():
            self.class_b_var.set(bundle.species[-1] if bundle.species[-1] != self.class_a_var.get() else bundle.species[0])
        self.last_result = None
        self.last_signature = None
        self._refresh_sample_inputs()

    def _on_feature_changed(self, _event=None) -> None:
        self.last_result = None
        self.last_signature = None
        self._refresh_sample_inputs()
        self.prediction_var.set("Settings changed. Train the model again before classifying a sample.")

    def _refresh_sample_inputs(self) -> None:
        for child in self.sample_fields_frame.winfo_children():
            child.destroy()
        self.sample_fields = []
        selected_features = (self.feature_1_var.get(), self.feature_2_var.get())
        numeric_display = self.dataset.frame
        for index, feature in enumerate(selected_features):
            row = ttk.Frame(self.sample_fields_frame)
            row.grid(row=index, column=0, sticky="w", pady=5)
            ttk.Label(row, text=f"{feature}:", width=24).grid(row=0, column=0, sticky="w")
            if feature == "OriginLocation":
                options = list(self.dataset.origin_mapping.keys())
                default = options[0] if options else ""
                variable = tk.StringVar(value=default)
                widget = ttk.Combobox(row, textvariable=variable, values=options, state="readonly", width=24)
                kind = "location"
            else:
                usable = numeric_display[feature].dropna()
                default = f"{float(usable.median()):.4f}" if not usable.empty else "0"
                variable = tk.StringVar(value=default)
                widget = ttk.Entry(row, textvariable=variable, width=27)
                kind = "numeric"
            widget.grid(row=0, column=1, sticky="w", padx=(5, 0))
            self.sample_fields.append((feature, variable, kind))

    def _read_configuration(self) -> dict:
        self._reload_dataset_if_needed()
        feature_names = (self.feature_1_var.get(), self.feature_2_var.get())
        if feature_names[0] == feature_names[1]:
            raise ValueError("Please choose two different features.")
        class_a, class_b = self.class_a_var.get(), self.class_b_var.get()
        if class_a == class_b:
            raise ValueError("Please choose two different classes.")
        learning_rate = float(self.learning_rate_var.get())
        epochs = int(self.epochs_var.get())
        mse_threshold = float(self.mse_threshold_var.get())
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        if epochs < 1:
            raise ValueError("Epochs must be at least 1.")
        if mse_threshold < 0:
            raise ValueError("MSE threshold cannot be negative.")
        return {
            "feature_names": feature_names,
            "class_a": class_a,
            "class_b": class_b,
            "algorithm": self.algorithm_var.get(),
            "learning_rate": learning_rate,
            "epochs": epochs,
            "mse_threshold": mse_threshold,
            "use_bias": self.use_bias_var.get(),
        }

    @staticmethod
    def _signature(config: dict, dataset_path: Path) -> tuple:
        return (
            str(dataset_path.resolve()), config["feature_names"], config["class_a"], config["class_b"],
            config["algorithm"], config["learning_rate"], config["epochs"],
            config["mse_threshold"], config["use_bias"],
        )

    def _train_model(self) -> None:
        try:
            config = self._read_configuration()
            result = run_experiment(dataset=self.dataset, random_state=42, **config)
            self.last_result = result
            self.last_signature = self._signature(config, self.dataset.source_path)
            self.accuracy_var.set(f"{result.accuracy:.1%}")
            self.model_info_var.set(
                f"{result.algorithm} | features: {result.feature_names[0]} + {result.feature_names[1]} | "
                f"train: {len(result.y_train)} (30/class) | test: {len(result.y_test)} (20/class) | "
                f"epochs used: {result.model.n_epochs_}/{result.epochs_requested} | "
                f"final train MSE: {result.final_mse:.4f} | bias: {'on' if result.use_bias else 'off'}"
            )
            matrix = result.confusion_matrix
            self.confusion_var.set(
                "Confusion Matrix — rows = actual, columns = predicted\n"
                f"                  {result.class_a:>11} {result.class_b:>11}\n"
                f"Actual {result.class_a:<10} {matrix[0, 0]:>11} {matrix[0, 1]:>11}\n"
                f"Actual {result.class_b:<10} {matrix[1, 0]:>11} {matrix[1, 1]:>11}"
            )
            self._show_plot(result)
            self.prediction_var.set("Model trained. Enter a sample using the same selected features to classify it.")
            self.status_var.set("Training and evaluation completed. Confusion matrix and accuracy were calculated manually.")
        except Exception as exc:
            messagebox.showerror("Training failed", str(exc), parent=self.root)
            self.status_var.set(f"Training failed: {exc}")

    def _show_plot(self, result: ExperimentResult) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.figure is not None:
            plt.close(self.figure)
        self.figure = create_decision_boundary_figure(result, self.dataset.origin_mapping)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _predict_sample(self) -> None:
        try:
            if self.last_result is None or self.last_signature is None:
                raise ValueError("Train a model first.")
            config = self._read_configuration()
            signature = self._signature(config, self.dataset.source_path)
            if signature != self.last_signature:
                raise ValueError("The settings have changed since training. Train the model again first.")

            values: list[float] = []
            for feature, variable, kind in self.sample_fields:
                if kind == "location":
                    selected_location = variable.get()
                    if selected_location not in self.dataset.origin_mapping:
                        raise ValueError(f"Choose a valid OriginLocation for {feature}.")
                    values.append(float(self.dataset.origin_mapping[selected_location]))
                else:
                    try:
                        values.append(float(variable.get()))
                    except ValueError as exc:
                        raise ValueError(f"Enter a valid numeric value for {feature}.") from exc
            prediction = classify_sample(self.last_result, values)
            self.prediction_var.set(
                f"Predicted class: {prediction.predicted_class}\n"
                f"Decision score: {prediction.decision_score:.4f} (not a probability)\n"
                f"Input: {self.last_result.feature_names[0]}={prediction.input_values[0]:.4f}, "
                f"{self.last_result.feature_names[1]}={prediction.input_values[1]:.4f}"
            )
            self.status_var.set(f"Single sample classified as {prediction.predicted_class}.")
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc), parent=self.root)

    def _generate_report(self) -> None:
        try:
            self._reload_dataset_if_needed()
            project_dir = Path(__file__).resolve().parent
            output_dir = project_dir / "output"
            self.generate_report_button.configure(state="disabled")
            self.report_status_var.set("Running all experiments and saving plots…")
            self.root.update_idletasks()
            report_path = generate_analysis_report(self.dataset.source_path, output_dir)
            self.report_status_var.set(
                f"Markdown report: {report_path}\n"
                f"HTML report: {output_dir / 'analysis_report.html'}\n"
                f"Results table: {output_dir / 'experiment_results.csv'}\n"
                f"Plots folder: {output_dir / 'figures'}"
            )
            self.open_report_button.configure(state="normal")
            self.status_var.set("Full report generated successfully.")
            messagebox.showinfo("Report generated", f"Analysis report saved to:\n{report_path}", parent=self.root)
        except Exception as exc:
            self.report_status_var.set(f"Report generation failed: {exc}")
            messagebox.showerror("Report generation failed", str(exc), parent=self.root)
        finally:
            self.generate_report_button.configure(state="normal")

    def _open_report(self) -> None:
        report_path = Path(__file__).resolve().parent / "output" / "analysis_report.html"
        if report_path.is_file():
            webbrowser.open(report_path.as_uri())
        else:
            messagebox.showwarning("Report not found", "Generate the report first.", parent=self.root)


def launch_gui() -> None:
    root = tk.Tk()
    PenguinsClassifierGUI(root)
    root.mainloop()
