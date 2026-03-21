# from xgboost import XGBClassifier
# from scipy.sparse import hstack
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# class Trainer:
#     def __init__(self):

#         # 🔹 TF-IDF models (keep strong)
#         self.model_state_tfidf = XGBClassifier(
#             n_estimators=150,
#             max_depth=5,
#             learning_rate=0.1,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             eval_metric="mlogloss"
#         )

#         self.model_intensity_tfidf = XGBClassifier(
#             n_estimators=150,
#             max_depth=5,
#             learning_rate=0.1,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             eval_metric="mlogloss"
#         )

#         # 🔹 BERT models (SIMPLER = better)
#         self.model_state_bert = XGBClassifier(
#             n_estimators=100,
#             max_depth=3,            # 🔥 reduced
#             learning_rate=0.05,     # 🔥 smoother learning
#             subsample=0.8,
#             colsample_bytree=0.8,
#             eval_metric="mlogloss"
#         )

#         self.model_intensity_bert = XGBClassifier(
#             n_estimators=100,
#             max_depth=3,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             eval_metric="mlogloss"
#         )

#     def train(self, X_text_tfidf, X_text_bert, X_meta, y_state, y_intensity):

#         # =========================
#         # 🔹 TF-IDF TRAINING
#         # =========================
#         X_tfidf = hstack([X_text_tfidf, X_meta])

#         classes = np.unique(y_state)
#         class_weights = compute_class_weight(
#             class_weight="balanced",
#             classes=classes,
#             y=y_state
#         )

#         weights_dict = dict(zip(classes, class_weights))
#         sample_weights = np.array([weights_dict[y] for y in y_state])

#         self.model_state_tfidf.fit(X_tfidf, y_state, sample_weight=sample_weights)
#         self.model_intensity_tfidf.fit(X_tfidf, y_intensity)

#         # =========================
#         # 🔹 BERT TRAINING
#         # =========================
#         X_meta_dense = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta
#         X_bert = np.hstack([X_text_bert, X_meta_dense])

#         self.model_state_bert.fit(X_bert, y_state, sample_weight=sample_weights)
#         self.model_intensity_bert.fit(X_bert, y_intensity)

#         return (
#             self.model_state_tfidf,
#             self.model_intensity_tfidf,
#             self.model_state_bert,
#             self.model_intensity_bert
#         )



from xgboost import XGBClassifier
from scipy.sparse import hstack
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import numpy as np


class Trainer:
    def __init__(self):

        # =========================
        # 🔹 TF-IDF MODELS (STRONG + REGULARIZED)
        # =========================
        self.model_state_tfidf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,          # 🔥 L1 regularization
            reg_lambda=1.5,         # 🔥 L2 regularization
            eval_metric="mlogloss",
            use_label_encoder=False
        )

        self.model_intensity_tfidf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            eval_metric="mlogloss",
            use_label_encoder=False
        )

        # =========================
        # 🔹 BERT MODELS (SIMPLER + STABLE)
        # =========================
        self.model_state_bert = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            use_label_encoder=False
        )

        self.model_intensity_bert = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            use_label_encoder=False
        )

    def _get_sample_weights(self, y):
        classes = np.unique(y)

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y
        )

        weights_dict = dict(zip(classes, class_weights))
        return np.array([weights_dict[label] for label in y])

    def train(self, X_text_tfidf, X_text_bert, X_meta, y_state, y_intensity):

        # =========================
        # 🔹 TF-IDF TRAINING
        # =========================
        X_tfidf = hstack([X_text_tfidf, X_meta])

        state_weights = self._get_sample_weights(y_state)
        intensity_weights = self._get_sample_weights(y_intensity)

        # Train base models
        self.model_state_tfidf.fit(X_tfidf, y_state, sample_weight=state_weights)
        self.model_intensity_tfidf.fit(X_tfidf, y_intensity, sample_weight=intensity_weights)

        # 🔥 CALIBRATION (IMPORTANT)
        self.model_state_tfidf = CalibratedClassifierCV(
            self.model_state_tfidf, method='sigmoid', cv=3
        )
        self.model_state_tfidf.fit(X_tfidf, y_state)

        # =========================
        # 🔹 BERT TRAINING
        # =========================
        X_meta_dense = X_meta.toarray() if hasattr(X_meta, "toarray") else X_meta
        X_bert = np.hstack([X_text_bert, X_meta_dense])

        self.model_state_bert.fit(X_bert, y_state, sample_weight=state_weights)
        self.model_intensity_bert.fit(X_bert, y_intensity, sample_weight=intensity_weights)

        # 🔥 CALIBRATION (BERT)
        self.model_state_bert = CalibratedClassifierCV(
            self.model_state_bert, method='sigmoid', cv=3
        )
        self.model_state_bert.fit(X_bert, y_state)

        return (
            self.model_state_tfidf,
            self.model_intensity_tfidf,
            self.model_state_bert,
            self.model_intensity_bert
        )