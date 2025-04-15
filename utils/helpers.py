
import pandas as pd

def load_labels(label_file):
    df = pd.read_csv(label_file)
    return df[['filename', 'label']]
