```markdown
# Dialogue Act Recognition - Speech Understanding Course Project

This repository contains experiments and implementations for **Dialogue Act Recognition** across multiple approaches, including context-aware self-attention, classical NLP pipelines, Fourier-based transformers, generative AI summaries, and speech-to-text demos.

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

Dialogue Act Recognition (DAR) aims to classify utterances in a conversation into functional categories (e.g., question, statement, backchannel). This repo showcases:

- **Neural models** leveraging contextual information (self-attention, RNNs)
- **Traditional pipelines** with feature extraction (text/speech) and classical classifiers (XGBoost)
- **Fourier-based Transformer (FNet)** for lightweight sequence modeling
- **Generative AI** methods for summarization of speech dialogues
- **Speech-to-Text** integration demos

## Features

- Plug-and-play experiments in `exps/`
- Modular code for data loading, feature extraction, modeling, and evaluation
- Pretrained/saved models in `dialogue_act_recognition/saved_models/`
- Sample results in `dialogue_act_recognition/results/`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/majisouvik26-dialogue_act_recognition.git
   cd majisouvik26-dialogue_act_recognition
   ```
2. Create a virtual environment and activate:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies for the core pipeline:
   ```bash
   pip install -r exps/dialogue_act_recognition/requirements.txt
   ```
4. (Optional) Install additional packages for other experiments:
   ```bash
   pip install torch transformers librosa scikit-learn numpy pandas
   ```

## Usage

Each experiment under `exps/` has its own README and instructions. Below are quick start guides:

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

Open the Jupyter notebook:

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
majisouvik26-dialogue_act_recognition/
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

---
