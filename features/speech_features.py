import parselmouth
import numpy as np
import os
import pandas as pd

def extract_features_from_file(filepath):
    snd = parselmouth.Sound(filepath)

    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    features = {
        'mean_pitch': np.mean(pitch.selected_array['frequency']),
        'std_pitch': np.std(pitch.selected_array['frequency']),
        'mean_intensity': np.mean(intensity.values.T),
        'std_intensity': np.std(intensity.values.T),
        'jitter_local': parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
        'shimmer_local': parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        'hnr': parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0).values.T.mean()
    }
    return features

def extract_features_from_directory(directory):
    all_features = []
    filenames = []

    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            features = extract_features_from_file(filepath)
            all_features.append(features)
            filenames.append(filename)

    df = pd.DataFrame(all_features)
    df['filename'] = filenames
    return df
