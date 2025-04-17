# Dialogue Act Recognition - Speech Understanding Course Project

This repository contains implementations and experiments for **Dialogue Act Recognition** using diverse techniques, including context-aware self-attention, classical NLP pipelines, Fourier-based transformers, generative AI summaries, and speech-to-text demos.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Context-Aware Self-Attention](#context-aware-self-attention)
  - [Dialogue Act Recognition Pipeline](#dialogue-act-recognition-pipeline)
  - [Fourier-Based Transformer](#fourier-based-transformer)
  - [Generative AI Summarization](#generative-ai-summarization)
  - [Speech-to-Text Demo](#speech-to-text-demo)
- [Directory Structure](#directory-structure)
- [License](#license)

## Project Overview

**Dialogue Act Recognition (DAR)** involves classifying utterances in a conversation into functional categories such as questions, statements, or backchannels. This project explores:

- **Neural models** using contextual attention and RNNs
- **Classical pipelines** with handcrafted features and models like XGBoost
- **FNet (Fourier-based Transformer)** for efficient sequence modeling
- **Generative AI** for summarizing spoken dialogues
- **Speech-to-Text** demos for real-time processing

## Features

- Modular experiments under `exps/`
- Clean separation of data loading, modeling, training, and evaluation
- Pretrained models in `saved_models/`
- Evaluation results in `results/`

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/majisouvik26/dialogue_act_recognition.git
   cd dialogue_act_recognition
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install core dependencies**
   ```bash
   pip install -r exps/dialogue_act_recognition/requirements.txt
   ```

4. **(Optional) Install additional dependencies for all experiments**
   ```bash
   pip install torch transformers librosa scikit-learn numpy pandas
   ```

## Usage

Each subdirectory in `exps/` contains its own README or scripts. Below are example usages:

### Context-Aware Self-Attention

```bash
cd exps/context_aware_self_attention
python main.py --config config.json
```

### Dialogue Act Recognition Pipeline

```bash
cd exps/dialogue_act_recognition
python train_pipeline.py
python predict.py --model saved_models/deep_model.h5 --input "Your utterance here"
```

### Fourier-Based Transformer

```bash
cd exps/fourier
python train.py --epochs 20 --batch_size 32
```

### Generative AI Summarization

```bash
cd exps/gen_ai
jupyter notebook gen_ai_summarization_speech.ipynb
```

### Speech-to-Text Demo

```bash
cd exps/speech_to_text
python demo.py --audio path/to/audio.wav
```

## Directory Structure

```
Directory structure:
└── majisouvik26-dialogue_act_recognition.git/
    ├── README.md
    ├── LICENSE
    └── exps/
        ├── context_aware_self_attention/
        │   ├── config.json
        │   ├── main.py
        │   ├── trainer.py
        │   ├── data/
        │   │   └── dataset.py
        │   └── models/
        │       ├── ContextAwareAttention.py
        │       ├── ContextAwareDAC.py
        │       ├── ConversationRNN.py
        │       └── UtteranceRNN.py
        ├── dialogue_act_recognition/
        │   ├── README.md
        │   ├── main.py
        │   ├── predict.py
        │   ├── requirements.txt
        │   ├── data/
        │   │   ├── __init__.py
        │   │   ├── labels.csv
        │   │   ├── loader.py
        │   │   └── transcripts.csv
        │   ├── features/
        │   │   ├── __init__.py
        │   │   ├── speech_features.py
        │   │   └── text_features.py
        │   ├── models/
        │   │   ├── __init__.py
        │   │   ├── deep_model.py
        │   │   └── xgboost_model.py
        │   ├── pipeline/
        │   │   ├── __init__.py
        │   │   ├── evaluate.py
        │   │   └── train_pipeline.py
        │   ├── results/
        │   │   ├── deep_results.txt
        │   │   └── xgboost_results.txt
        │   ├── saved_models/
        │   │   ├── deep_model.h5
        │   │   └── xgb_model.json
        │   └── utils/
        │       ├── __init__.py
        │       └── helpers.py
        ├── fourier/
        │   ├── dataset.py
        │   ├── fnet_model.py
        │   └── train.py
        ├── gen_ai/
        │   └── gen_ai_summarization_speech.ipynb
        └── speech_to_text/
            └── demo.py

```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
