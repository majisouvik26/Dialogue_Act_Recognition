import pandas as pd

def load_transcripts(path):
    return pd.read_csv(path)

def load_labels(path):
    df = pd.read_csv(path)
    return df[['filename', 'label']]
