import argparse
import json
from config import OUTPUT_DIR
from src.data_loader import load_dataset
from src.training import train_and_evaluate, load_trained_model
from src.prediction import predict_single
from src.visualization import save_evaluation_plots, save_feature_distributions

def run_training():
    df = load_dataset()
    model, results, test_data = train_and_evaluate(df)
    X_test, y_test, metrics = test_data
    save_evaluation_plots(model, X_test, y_test, OUTPUT_DIR)
    save_feature_distributions(df, OUTPUT_DIR)
    print("\nTraining complete")
    print("Selected model:", results["best_model"])
    print(json.dumps(results["test_metrics"], indent=2))
    print("Charts saved in outputs/ and model saved in models/")

def run_prediction(args):
    model = load_trained_model()
    values = {
        "Pregnancies": args.pregnancies,
        "Glucose": args.glucose,
        "BloodPressure": args.blood_pressure,
        "SkinThickness": args.skin_thickness,
        "Insulin": args.insulin,
        "BMI": args.bmi,
        "DiabetesPedigreeFunction": args.pedigree,
        "Age": args.age,
    }
    result = predict_single(model, values)
    print(json.dumps(result, indent=2))
    print("This is an educational model output, not a medical diagnosis.")

def main():
    parser = argparse.ArgumentParser(description="Diabetes Prediction System")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train, compare and evaluate ML models")
    pred = sub.add_parser("predict", help="Predict from supplied measurements")
    pred.add_argument("--pregnancies", type=float, required=True)
    pred.add_argument("--glucose", type=float, required=True)
    pred.add_argument("--blood-pressure", type=float, required=True)
    pred.add_argument("--skin-thickness", type=float, required=True)
    pred.add_argument("--insulin", type=float, required=True)
    pred.add_argument("--bmi", type=float, required=True)
    pred.add_argument("--pedigree", type=float, required=True)
    pred.add_argument("--age", type=float, required=True)
    args = parser.parse_args()
    if args.command == "train":
        run_training()
    else:
        run_prediction(args)

if __name__ == "__main__":
    main()
