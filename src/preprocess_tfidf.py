import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self):
        # 🔥 UPDATED: n-grams added
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.label_encoders = {}    

    def clean_text(self, text_series):
        return text_series.fillna("").str.lower()

    def fit_transform(self, df, text_col, meta_cols):
        text = self.clean_text(df[text_col])
        X_text = self.tfidf.fit_transform(text)

        X_meta = df[meta_cols].copy()

        numeric_cols = ['sleep_hours', 'stress_level', 'energy_level', 'duration_min']
        categorical_cols = [col for col in meta_cols if col not in numeric_cols]

        # numeric
        for col in numeric_cols:
            if col in X_meta.columns:
                X_meta[col] = pd.to_numeric(X_meta[col], errors='coerce')
                X_meta[col] = X_meta[col].fillna(X_meta[col].mean())

        # categorical
        for col in categorical_cols:
            X_meta[col] = X_meta[col].fillna("unknown").astype(str)

            le = LabelEncoder()
            X_meta[col] = le.fit_transform(X_meta[col])
            self.label_encoders[col] = le

        X_meta = X_meta.astype(float)

        return X_text, X_meta

    def transform(self, df, text_col, meta_cols):
        text = self.clean_text(df[text_col])
        X_text = self.tfidf.transform(text)

        X_meta = df[meta_cols].copy()

        numeric_cols = ['sleep_hours', 'stress_level', 'energy_level', 'duration_min']

        for col in numeric_cols:
            if col in X_meta.columns:
                X_meta[col] = pd.to_numeric(X_meta[col], errors='coerce')
                X_meta[col] = X_meta[col].fillna(X_meta[col].mean())

        for col in self.label_encoders.keys():
            X_meta[col] = X_meta[col].fillna("unknown").astype(str)

            le = self.label_encoders[col]

            X_meta[col] = X_meta[col].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )

            X_meta[col] = le.transform(X_meta[col])

        X_meta = X_meta.astype(float)

        return X_text, X_meta