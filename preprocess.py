import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_beats(record_name, label_map):
    record = wfdb.rdrecord(record_name, sampfrom=0, channels=[0])
    annotation = wfdb.rdann(record_name, 'atr')
    signal = record.p_signal.flatten()
    beats = []
    labels = []

    for i, sym in enumerate(annotation.symbol):
        if sym in label_map:
            idx = annotation.sample[i]
            beat = signal[idx-150:idx+150]
            if len(beat) == 300:
                beats.append(beat)
                labels.append(label_map[sym])
    return beats, labels

label_map = {'N': 0, 'L': 0, 'R': 0, 'A': 1, 'V': 1}
records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109']

all_beats = []
all_labels = []

for rec in records:
    beats, labels = extract_beats(f"mitdb/{rec}", label_map)
    all_beats.extend(beats)
    all_labels.extend(labels)

df = pd.DataFrame(all_beats)
df['label'] = all_labels
df.to_csv("mitbih_csv/mitbih_beats_binary.csv", index=False)
print("✅ تم حفظ البيانات المعالجة إلى mitbih_csv/mitbih_beats_binary.csv")
