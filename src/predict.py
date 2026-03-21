from scipy.sparse import hstack
import numpy as np


class Predictor:
    def __init__(
        self,
        model_state_tfidf,
        model_intensity_tfidf,
        model_state_bert,
        model_intensity_bert
    ):
        self.model_state_tfidf = model_state_tfidf
        self.model_intensity_tfidf = model_intensity_tfidf

        self.model_state_bert = model_state_bert
        self.model_intensity_bert = model_intensity_bert

    # =========================
    # 🔹 TF-IDF PIPELINE
    # =========================
    def _predict_tfidf(self, X_text, X_meta):

        X = hstack([X_text, X_meta])

        # ---- STATE ----
        state_probs = self.model_state_tfidf.predict_proba(X)
        state_pred = np.argmax(state_probs, axis=1)
        state_conf = np.max(state_probs, axis=1)

        # ---- INTENSITY ----
        intensity_probs = self.model_intensity_tfidf.predict_proba(X)
        intensity_pred = np.argmax(intensity_probs, axis=1)

        # ---- UNCERTAINTY ----
        uncertain = (state_conf < 0.4).astype(int)   # slightly stricter

        return {
            "state": state_pred,
            "intensity": intensity_pred,
            "confidence": state_conf,
            "probs": state_probs,
            "uncertain": uncertain
        }

    # =========================
    # 🔹 BERT PIPELINE
    # =========================
    def _predict_bert(self, X_text_bert, X_meta):

        # convert sparse → dense if needed
        X_meta_dense = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta

        X = np.hstack([X_text_bert, X_meta_dense])

        # ---- STATE ----
        state_probs = self.model_state_bert.predict_proba(X)
        state_pred = np.argmax(state_probs, axis=1)
        state_conf = np.max(state_probs, axis=1)

        # ---- INTENSITY ----
        intensity_probs = self.model_intensity_bert.predict_proba(X)
        intensity_pred = np.argmax(intensity_probs, axis=1)

        # ---- UNCERTAINTY ----
        uncertain = (state_conf < 0.4).astype(int)

        return {
            "state": state_pred,
            "intensity": intensity_pred,
            "confidence": state_conf,
            "probs": state_probs,
            "uncertain": uncertain
        }

    # =========================
    # 🔥 MAIN PREDICT
    # =========================
    def predict(self, X_text_tfidf, X_text_bert, X_meta):

        tfidf_out = self._predict_tfidf(X_text_tfidf, X_meta)
        bert_out = self._predict_bert(X_text_bert, X_meta)

        return {
            "tfidf": tfidf_out,
            "bert": bert_out
        }