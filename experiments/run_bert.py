import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from src.preprocess_bert import PreprocessorBERT
from src.train import Trainer

train = pd.read_csv("data/train.csv")
train.columns = train.columns.str.lower()

text_col = "journal_text"

meta_cols = [
    'sleep_hours','stress_level','energy_level',
    'time_of_day','ambience_type',
    'previous_day_mood','face_emotion_hint',
    'reflection_quality'
]

y = train["emotional_state"]

pre = PreprocessorBERT()
X_text, X_meta = pre.fit_transform(train, text_col, meta_cols)

X = np.hstack([X_text, X_meta])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

trainer = Trainer()
model_state, _ = trainer.train(X_train, None, y_train, y_train)

preds = model_state.predict(X_val)

report = classification_report(y_val, preds)

print(report)

with open("outputs/bert_results.txt", "w") as f:
    f.write(report)