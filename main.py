
from pipeline.train_pipeline import run_pipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Dialogue Act Recognition Training Pipeline")

    parser.add_argument('--features', choices=['text', 'speech', 'both'], default='both',
                        help="Which features to use: 'text', 'speech', or 'both'")
    parser.add_argument('--model', choices=['xgb', 'deep'], default='xgb',
                        help="Which model to use: 'xgb' or 'deep'")
    parser.add_argument('--text_data', type=str, default='data/transcripts.csv',
                        help="Path to transcript CSV file")
    parser.add_argument('--audio_data', type=str, default='data/audio/',
                        help="Path to directory with audio (.wav) files")
    parser.add_argument('--labels', type=str, default='data/labels.csv',
                        help="Path to labels CSV file")

    args = parser.parse_args()

    run_pipeline(feature_type=args.features,
                 model_type=args.model,
                 text_data_path=args.text_data,
                 audio_data_path=args.audio_data,
                 label_file=args.labels)

if __name__ == '__main__':
    main()
