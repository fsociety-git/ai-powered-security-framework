import argparse
from models.anomaly_detector import train_anomaly_model, detect_anomaly
from models.vuln_predictor import predict_vuln
from utils.report_generator import generate_report

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Security Framework")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train ML model")
    train_parser.add_argument("--model", required=True, choices=["anomaly", "vuln"])
    train_parser.add_argument("--data", required=True, help="Data path")

    predict_parser = subparsers.add_parser("predict", help="Predict with model")
    predict_parser.add_argument("--model", required=True, choices=["anomaly", "vuln"])
    predict_parser.add_argument("--input", required=True, help="Input data/code")

    args = parser.parse_args()

    results = {}
    if args.command == "train":
        if args.model == "anomaly":
            model = train_anomaly_model(args.data)
            results["status"] = "Trained"
        # Add vuln training if expanded
    elif args.command == "predict":
        if args.model == "anomaly":
            results = detect_anomaly(args.input)
        elif args.model == "vuln":
            results = predict_vuln(args.input)
    
    generate_report(results)

if __name__ == "__main__":
    main()