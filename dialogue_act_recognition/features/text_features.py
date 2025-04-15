import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text_features(texts, max_features=100):
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(texts)
    return pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names_out()), vectorizer
