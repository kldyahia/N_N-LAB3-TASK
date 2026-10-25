# Penguins Binary Classifier — Neural Networks Lab 3

A modular Python implementation of the Lab 3 task. It implements the Perceptron and Adaline algorithms, provides a Tkinter GUI, calculates the confusion matrix manually, plots a two-dimensional decision boundary, and generates the requested analysis report.

## Project structure

```text
penguins_classifier/
├── main.py                 # Starts the GUI
├── gui.py                  # Tkinter interface only
├── data_loader.py          # CSV validation, location encoding, balanced split
├── preprocessing.py        # Training-only median imputation and scaling
├── models.py               # Perceptron and Adaline implementations
├── evaluation.py            # Manual confusion matrix and accuracy
├── experiment.py            # Connects data, preprocessing, training, evaluation
├── visualization.py         # Decision boundary and scatter plot
├── report_generator.py      # Runs all combinations and writes the report
├── generate_report.py       # CLI shortcut for report generation
├── data/penguins.csv        # Dataset supplied for the task
├── tests/test_core.py       # Core unit tests
└── output/                  # Generated report, CSV table, and PNG plots
```

The interface and algorithm code are deliberately separated: `gui.py` calls `experiment.py`; it does not implement training loops or confusion-matrix calculations itself.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- Tkinter (usually included with standard Python installations on Windows)

## Run on Windows (PowerShell)

Open PowerShell in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If PowerShell blocks the activation script, activate the environment using the method allowed by your machine's policy, or call `.venv\Scripts\python.exe` directly.

## Run the automatic report without opening the GUI

```powershell
python generate_report.py
```

Optional custom paths:

```powershell
python generate_report.py --dataset "data\penguins.csv" --output "output"
```

The script tests all 10 feature pairs and all 3 class pairs for **each** algorithm, then selects five representative feature pairs per algorithm across the observed accuracy ranking. It produces:

- `output/analysis_report.md` — editable written analysis, five plots per algorithm, confusion matrices, and best-performing combinations.
- `output/analysis_report.html` — browser-friendly report with styling and embedded plot links.
- `output/experiment_results.csv` — all 60 experiment rows (30 per algorithm).
- `output/figures/` — 10 PNG decision-boundary plots.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Lab 3 implementation choices

- Each selected class contributes 30 unique training samples and the remaining 20 test samples (60 training and 40 test samples total).
- The original CSV is loaded without dropping rows. The two rows containing missing numeric measurements are retained and missing selected features are imputed from the **training split's** median.
- `OriginLocation` is mapped to deterministic numeric codes so it remains one selectable feature and the required plot remains two-dimensional. The encoded number is a category identifier, not a meaningful ordinal distance. One-hot encoding is not used because it would expand that selected feature into multiple input dimensions.
- Both selected features are standardized from training data only. Test samples and the decision-boundary grid use the same fitted transformation.
- Target labels are Class A = -1 and Class B = +1.
- Perceptron performs sample-by-sample error updates. Adaline uses a linear output and stochastic gradient descent to minimize mean squared error (MSE); its training can stop when the MSE threshold is reached.
- No scikit-learn package or ready-made confusion-matrix utility is used. The confusion matrix and overall accuracy are calculated in `evaluation.py` with ordinary loops.
- The GUI prediction score is a linear decision score, not a probability estimate.

## GUI workflow

1. Choose two different features and two different species.
2. Choose Perceptron or Adaline, set the learning rate, maximum epochs, MSE threshold, and Bias option.
3. Click **Train & Evaluate** to see the test accuracy, confusion matrix, and decision boundary.
4. Open **Classify One Sample** to enter one new sample for the same trained feature pair and get the predicted species plus the decision score.
5. Open **Full Analysis Report** to generate the full 10-plot report and all-combinations CSV.

The application uses a fixed random seed (42) for repeatability. The reported scores are for that specific split; they may change with a different split or training configuration. The new-sample decision score is not a probability.
