import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from features.text_features import extract_text_features
from tensorflow.keras.models import load_model
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Predict and evaluate using saved model")
    parser.add_argument("--model", choices=["xgb", "deep"], required=True, help="Model type: xgb or deep")
    parser.add_argument("--features", choices=["text"], default="text", help="Feature type (currently only text is supported)")
    parser.add_argument("--text_data", type=str, default="data/transcripts.csv", help="Path to transcript CSV file")
    parser.add_argument("--labels", type=str, default="data/labels.csv", help="Path to labels CSV file")
    args = parser.parse_args()

    # Load and merge data
    transcripts = pd.read_csv(args.text_data)
    labels = pd.read_csv(args.labels)
    df = pd.merge(transcripts, labels, on="filename")

    # Extract features
    X, _ = extract_text_features(df['transcript'].tolist())

    # Encode labels
    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(df['label'])

    if args.model == "xgb":
        model = xgb.XGBClassifier()
        model.load_model("saved_models/xgb_model.json")
        y_pred = model.predict(X)

    elif args.model == "deep":
        model = load_model("saved_models/deep_model.h5")
        y_pred_probs = model.predict(X.values)
        y_pred = np.argmax(y_pred_probs, axis=1)

    else:
        raise ValueError("Unsupported model type")

    # Evaluate
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
