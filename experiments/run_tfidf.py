import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack

from src.preprocess_tfidf import Preprocessor
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

pre = Preprocessor()
X_text, X_meta = pre.fit_transform(train, text_col, meta_cols)

X_train, X_val, y_train, y_val = train_test_split(X_text, y, test_size=0.2)

trainer = Trainer()
model_state, _ = trainer.train(X_train, X_meta[:len(X_train)], y_train, y_train)

X_val_final = hstack([X_val, X_meta[len(X_train):]])

preds = model_state.predict(X_val_final)

report = classification_report(y_val, preds)

print(report)

with open("outputs/tfidf_results.txt", "w") as f:
    f.write(report)