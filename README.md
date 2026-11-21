# GlucoSense — Diabetes Prediction System

**Author:** Harsh Saini  
**Stack:** Python, Pandas, Scikit-learn, Streamlit, Plotly

A modular machine-learning project that compares classification algorithms on the Pima Indians Diabetes dataset and provides both a command-line workflow and a separate Streamlit dashboard.

## Project structure

```text
dsp/
├── app.py                  # Streamlit interface only
├── main.py                 # CLI entry point / workflow orchestration
├── config.py               # Paths and project configuration
├── requirements.txt
├── README.md
├── data/
│   └── diabetes.csv        # Downloaded automatically on first run
├── models/
│   └── best_model.joblib   # Saved after training
├── outputs/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── correlation_heatmap.png
└── src/
    ├── data_loader.py      # Dataset loading and validation
    ├── preprocessing.py    # Missing-value handling and scaling
    ├── models.py           # Model definitions
    ├── training.py         # Cross-validation, model selection, persistence
    ├── evaluation.py      # Classification metrics
    ├── prediction.py      # Single-record prediction
    └── visualization.py   # Evaluation charts
```

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run the machine-learning pipeline (without GUI)

```bash
python main.py train
```

This downloads the dataset if needed, preprocesses the data, compares Logistic Regression, Random Forest, Extra Trees and Histogram Gradient Boosting using stratified 5-fold cross-validation, evaluates the selected model on a held-out test set, and saves the model and reports.

## Make a prediction from the command line

First run training, then:

```bash
python main.py predict --pregnancies 2 --glucose 130 --blood-pressure 70 --skin-thickness 25 --insulin 90 --bmi 28.5 --pedigree 0.45 --age 35
```

## Run the separate dashboard

```bash
python -m streamlit run app.py
```

`app.py` contains presentation and user interaction only. Model logic is imported from `src/`; it is not duplicated in the interface.

## Dataset

The project uses the Pima Indians Diabetes dataset, a commonly used educational benchmark with eight input features and a binary outcome. On first run, the CSV is fetched from the Plotly datasets GitHub repository and cached locally in `data/`. Internet access is required for the first download unless the CSV is supplied manually.

## Important notes on model quality

- There is no honest way to guarantee a particular accuracy before evaluating the model on data.
- The project selects a model using mean cross-validation ROC-AUC, then reports metrics on a separate hold-out test set.
- Accuracy alone can be misleading when classes are imbalanced; review recall, precision, F1 and ROC-AUC together.
- This dataset is limited and does not represent every population. Results are for academic demonstration only, not clinical diagnosis or treatment.
