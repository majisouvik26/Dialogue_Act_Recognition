from features.text_features import extract_text_features
from features.speech_features import extract_features_from_directory
from models.xgboost_model import train_xgboost
from models.deep_model import train_deep_model
from data.loader import load_labels, load_transcripts
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def run_pipeline(feature_type='both', model_type='xgb', 
                 text_data_path='data/transcripts.csv', 
                 audio_data_path='data/audio/', 
                 label_file='data/labels.csv'):
    
    print(f"Running pipeline with {feature_type} features using {model_type} model.")

    labels = load_labels(label_file)

    if feature_type in ['text', 'both']:
        df_text = load_transcripts(text_data_path)
        X_text, vectorizer = extract_text_features(df_text['transcript'].tolist())
        X_text['filename'] = df_text['filename']

    if feature_type in ['speech', 'both']:
        X_speech = extract_features_from_directory(audio_data_path)

    if feature_type == 'text':
        X = pd.merge(X_text, labels, on='filename')
    elif feature_type == 'speech':
        X = pd.merge(X_speech, labels, on='filename')
    else:
        merged = pd.merge(X_speech, X_text, on='filename')
        X = pd.merge(merged, labels, on='filename')

    y = X['label']
    X.drop(columns=['filename', 'label'], inplace=True)

    if model_type == 'xgb':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        model, acc, report = train_xgboost(X, y_encoded)
    elif model_type == 'deep':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        model, acc, report = train_deep_model(X.values, y_encoded)
    else:
        raise ValueError("Unsupported model_type. Choose 'xgb' or 'deep'.")

    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification Report:\n", report)
