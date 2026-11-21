from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
DATA_PATH = DATA_DIR / "diabetes.csv"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Pima Indians Diabetes dataset (768 rows; commonly used educational benchmark).
DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

TARGET = "Outcome"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
