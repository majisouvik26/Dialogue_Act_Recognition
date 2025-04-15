# Dialogue Act Recognition (Text & Speech)

This project performs dialogue act recognition using machine learning models on the **Switchboard Dialog Act Corpus**, classifying utterances into their functional types (e.g., question, statement, backchannel). It supports both **text-based** and **acoustic-based** features, and two model types: **XGBoost** and **Deep Learning** (Keras).

---

## Project Structure

```
dialogue_act_recognition/
├── data/                   # Dataset: transcripts, labels, (optionally audio)
├── features/               # Feature extraction modules (text, speech)
├── models/                 # Model training modules (XGBoost, Deep Learning)
├── pipeline/               # Training and evaluation pipeline
├── saved_models/           # Saved models and label encoders
├── utils/                  # Helpers and data utilities
├── main.py                 # CLI entry point for training
├── predict.py              # CLI tool to load saved model and evaluate
├── requirements.txt
└── README.md
```

---

## Dependencies

Install all required packages with:

```bash
pip install -r requirements.txt
```

> If you're using speech features, also install:
```bash
pip install praat-parselmouth
```

---

## Dataset Format

Prepare the dataset in CSV format inside the `data/` folder:

### `transcripts.csv`
| filename    | transcript             |
|-------------|-------------------------|
| utt_0001.wav | I just got back from work. |
| utt_0002.wav | What do you mean?        |

### `labels.csv`
| filename    | label     |
|-------------|-----------|
| utt_0001.wav | sd        |
| utt_0002.wav | qw        |

---

## Training

### XGBoost (Text only)
```bash
python main.py --features text --model xgb --text_data data/transcripts.csv --labels data/labels.csv
```

### Deep Learning (Text only)
```bash
python main.py --features text --model deep --text_data data/transcripts.csv --labels data/labels.csv
```

Trained models are saved in `saved_models/`.

---

## Evaluation / Prediction

Use the provided `predict.py` to evaluate saved models:

### For XGBoost:
```bash
python predict.py --model xgb --features text
```

### For Deep Learning:
```bash
python predict.py --model deep --features text
```

This will output:
- Accuracy
- Classification Report
- Confusion Matrix (visual)

---

## Features

- Text features via TF-IDF
- Speech features via Parselmouth (if audio is provided)
- XGBoost or Deep Learning model choice
- Modular pipeline for easy extension
- CLI training and evaluation
- Visualization via confusion matrix

---

## Future Improvements

- Add inference for single utterances
- Integrate LIWC or Empath-based linguistic features
- Improve handling of class imbalance
- Add audio feature selection and augmentation

---

## License

This project is for academic and research purposes. The Switchboard dataset is licensed — please ensure you have access to it via [LDC](https://catalog.ldc.upenn.edu/LDC97T03).

---

## Credits

Created as part of a dialogue act classification project inspired by [SwDA Corpus](https://catalog.ldc.upenn.edu/LDC97T03).