# # # # # import pandas as pd
# # # # # from src.preprocess import Preprocessor
# # # # # from src.train import Trainer
# # # # # from src.predict import Predictor
# # # # # from src.decision import decide_action

# # # # # # Load data
# # # # # train = pd.read_csv("data/train.csv")
# # # # # test = pd.read_csv("data/test.csv")


# # # # # # Columns
# # # # # text_col = "journal_text"

# # # # # meta_cols = [
# # # # #     'sleep_hours', 'stress_level', 'energy_level',
# # # # #     'time_of_day', 'ambience_type',
# # # # #     'previous_day_mood', 'face_emotion_hint',
# # # # #     'reflection_quality'
# # # # # ]

# # # # # # Targets
# # # # # y_state = train["emotional_state"]
# # # # # y_intensity = train["intensity"]

# # # # # # Preprocess
# # # # # pre = Preprocessor()
# # # # # X_text_train, X_meta_train = pre.fit_transform(train, text_col, meta_cols)
# # # # # X_text_test, X_meta_test = pre.transform(test, text_col, meta_cols)

# # # # # # Train
# # # # # trainer = Trainer()
# # # # # model_state, model_intensity = trainer.train(
# # # # #     X_text_train, X_meta_train, y_state, y_intensity
# # # # # )

# # # # # # Predict
# # # # # predictor = Predictor(model_state, model_intensity)
# # # # # pred_state, pred_intensity, confidence, uncertain_flag = predictor.predict(
# # # # #     X_text_test, X_meta_test
# # # # # )

# # # # # # Decision
# # # # # actions = []
# # # # # timings = []

# # # # # for i in range(len(test)):
# # # # #     action, timing = decide_action(
# # # # #         pred_state[i],
# # # # #         pred_intensity[i],
# # # # #         test['stress_level'].iloc[i],
# # # # #         test['energy_level'].iloc[i],
# # # # #         test['time_of_day'].iloc[i]
# # # # #     )
# # # # #     actions.append(action)
# # # # #     timings.append(timing)

# # # # # # Save output
# # # # # output = pd.DataFrame({
# # # # #     "id": test["id"],
# # # # #     "predicted_state": pred_state,
# # # # #     "predicted_intensity": pred_intensity,
# # # # #     "confidence": confidence,
# # # # #     "uncertain_flag": uncertain_flag,
# # # # #     "what_to_do": actions,
# # # # #     "when_to_do": timings
# # # # # })

# # # # # output.to_csv("outputs/predictions.csv", index=False)

# # # # # print("✅ predictions.csv generated!")

# # # # import pandas as pd
# # # # from sklearn.model_selection import train_test_split
# # # # from sklearn.metrics import classification_report
# # # # from scipy.sparse import hstack

# # # # from src.preprocess import Preprocessor
# # # # from src.train import Trainer
# # # # from src.predict import Predictor
# # # # from src.decision import decide_action

# # # # # ---- Load data ----
# # # # train = pd.read_csv("data/train.csv")
# # # # test = pd.read_csv("data/test.csv")

# # # # # ---- Normalize column names ----
# # # # train.columns = train.columns.str.strip().str.lower()
# # # # test.columns = test.columns.str.strip().str.lower()

# # # # # ---- Columns ----
# # # # text_col = "journal_text"

# # # # meta_cols = [
# # # #     'sleep_hours',
# # # #     'stress_level',
# # # #     'energy_level',
# # # #     'time_of_day',
# # # #     'ambience_type',
# # # #     'previous_day_mood',
# # # #     'face_emotion_hint',
# # # #     'reflection_quality'
# # # # ]

# # # # # ---- Targets ----
# # # # y_state = train["emotional_state"]
# # # # y_intensity = train["intensity"]

# # # # # ---- Preprocess ----
# # # # pre = Preprocessor()
# # # # X_text, X_meta = pre.fit_transform(train, text_col, meta_cols)

# # # # # ---- Train / Validation Split (FIXED) ----
# # # # X_text_tr, X_text_val, X_meta_tr, X_meta_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
# # # #     X_text,
# # # #     X_meta,
# # # #     y_state,
# # # #     y_intensity,
# # # #     test_size=0.2,
# # # #     random_state=42
# # # # )

# # # # # ---- Train ----
# # # # trainer = Trainer()
# # # # model_state, model_intensity = trainer.train(
# # # #     X_text_tr,
# # # #     X_meta_tr,
# # # #     y_state_tr,
# # # #     y_int_tr
# # # # )

# # # # # ---- Validate ----
# # # # X_val = hstack([X_text_val, X_meta_val])
# # # # val_preds = model_state.predict(X_val)

# # # # print("\n📊 Validation Report (Emotional State):")
# # # # print(classification_report(y_state_val, val_preds))

# # # # # ---- Prepare Test Data ----
# # # # X_text_test, X_meta_test = pre.transform(test, text_col, meta_cols)

# # # # # ---- Predict ----
# # # # predictor = Predictor(model_state, model_intensity)
# # # # pred_state, pred_intensity, confidence, uncertain_flag = predictor.predict(
# # # #     X_text_test,
# # # #     X_meta_test
# # # # )

# # # # # ---- Decision Engine ----
# # # # actions = []
# # # # timings = []

# # # # for i in range(len(test)):
# # # #     action, timing = decide_action(
# # # #         pred_state[i],
# # # #         pred_intensity[i],
# # # #         test['stress_level'].iloc[i],
# # # #         test['energy_level'].iloc[i],
# # # #         test['time_of_day'].iloc[i]
# # # #     )
# # # #     actions.append(action)
# # # #     timings.append(timing)

# # # # # ---- Save Output ----
# # # # output = pd.DataFrame({
# # # #     "id": test["id"],
# # # #     "predicted_state": pred_state,
# # # #     "predicted_intensity": pred_intensity,
# # # #     "confidence": confidence,
# # # #     "uncertain_flag": uncertain_flag,
# # # #     "what_to_do": actions,
# # # #     "when_to_do": timings
# # # # })

# # # # output.to_csv("outputs/predictions.csv", index=False)

# # # # print("\n✅ predictions.csv generated!")

# # # import pandas as pd
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.metrics import classification_report
# # # from sklearn.preprocessing import LabelEncoder
# # # from scipy.sparse import hstack

# # # from preprocess_tfidf import Preprocessor
# # # from src.train import Trainer
# # # from src.predict import Predictor
# # # from src.decision import decide_action

# # # # ---- Load data ----
# # # train = pd.read_csv("data/train.csv")
# # # test = pd.read_csv("data/test.csv")

# # # # ---- Normalize column names ----
# # # train.columns = train.columns.str.strip().str.lower()
# # # test.columns = test.columns.str.strip().str.lower()

# # # # ---- Columns ----
# # # text_col = "journal_text"

# # # meta_cols = [
# # #     'sleep_hours',
# # #     'stress_level',
# # #     'energy_level',
# # #     'time_of_day',
# # #     'ambience_type',
# # #     'previous_day_mood',
# # #     'face_emotion_hint',
# # #     'reflection_quality'
# # # ]

# # # # ---- Targets ----
# # # y_state = train["emotional_state"]
# # # y_intensity = train["intensity"]

# # # # ---- Encode labels (NEW for XGBoost) ----
# # # le_state = LabelEncoder()
# # # y_state_encoded = le_state.fit_transform(y_state)

# # # le_intensity = LabelEncoder()
# # # y_int_encoded = le_intensity.fit_transform(y_intensity)

# # # # ---- Preprocess ----
# # # pre = Preprocessor()
# # # X_text, X_meta = pre.fit_transform(train, text_col, meta_cols)

# # # # ---- Train / Validation Split ----
# # # X_text_tr, X_text_val, X_meta_tr, X_meta_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
# # #     X_text,
# # #     X_meta,
# # #     y_state_encoded,
# # #     y_int_encoded,
# # #     test_size=0.2,
# # #     random_state=42
# # # )

# # # # ---- Train ----
# # # trainer = Trainer()
# # # model_state, model_intensity = trainer.train(
# # #     X_text_tr,
# # #     X_meta_tr,
# # #     y_state_tr,
# # #     y_int_tr
# # # )

# # # # ---- Validate ----
# # # X_val = hstack([X_text_val, X_meta_val])
# # # val_preds = model_state.predict(X_val)

# # # # Decode predictions for readable report
# # # val_preds_decoded = le_state.inverse_transform(val_preds)
# # # y_state_val_decoded = le_state.inverse_transform(y_state_val)

# # # print("\n📊 Validation Report (Emotional State):")
# # # print(classification_report(y_state_val_decoded, val_preds_decoded))

# # # # ---- Prepare Test Data ----
# # # X_text_test, X_meta_test = pre.transform(test, text_col, meta_cols)

# # # # ---- Predict ----
# # # predictor = Predictor(model_state, model_intensity)
# # # pred_state, pred_intensity, confidence, uncertain_flag = predictor.predict(
# # #     X_text_test,
# # #     X_meta_test
# # # )

# # # # ---- Decode predictions ----
# # # pred_state = le_state.inverse_transform(pred_state)
# # # pred_intensity = le_intensity.inverse_transform(pred_intensity)

# # # # ---- Decision Engine ----
# # # actions = []
# # # timings = []

# # # for i in range(len(test)):
# # #     action, timing = decide_action(
# # #         pred_state[i],
# # #         pred_intensity[i],
# # #         test['stress_level'].iloc[i],
# # #         test['energy_level'].iloc[i],
# # #         test['time_of_day'].iloc[i]
# # #     )
# # #     actions.append(action)
# # #     timings.append(timing)

# # # # ---- Save Output ----
# # # output = pd.DataFrame({
# # #     "id": test["id"],
# # #     "predicted_state": pred_state,
# # #     "predicted_intensity": pred_intensity,
# # #     "confidence": confidence,
# # #     "uncertain_flag": uncertain_flag,
# # #     "what_to_do": actions,
# # #     "when_to_do": timings
# # # })

# # # output.to_csv("outputs/predictions.csv", index=False)

# # # print("\n✅ predictions.csv generated!")


# # # # ---- Manual Test ----
# # # sample = pd.DataFrame([{
# # #     "journal_text": "ok",
# # #     "sleep_hours": 6,
# # #     "stress_level": 3,
# # #     "energy_level": 3,
# # #     "time_of_day": "morning",
# # #     "ambience_type": "home",
# # #     "previous_day_mood": "neutral",
# # #     "face_emotion_hint": "neutral",
# # #     "reflection_quality": "low"
# # # }])

# # # X_text_sample, X_meta_sample = pre.transform(sample, text_col, meta_cols)

# # # pred_s, pred_i, conf, unc = predictor.predict(X_text_sample, X_meta_sample)

# # # pred_s = le_state.inverse_transform(pred_s)
# # # pred_i = le_intensity.inverse_transform(pred_i)

# # # action, timing = decide_action(
# # #     pred_s[0],
# # #     pred_i[0],
# # #     sample["stress_level"][0],
# # #     sample["energy_level"][0],
# # #     sample["time_of_day"][0],
# # #     sample["journal_text"][0]
# # # )

# # # print("\n--- SAMPLE TEST ---")
# # # print("Text:", sample["journal_text"][0])
# # # print("State:", pred_s[0])
# # # print("Intensity:", pred_i[0])
# # # print("Confidence:", conf[0])
# # # print("Action:", action)
# # # print("When:", timing)


# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import classification_report
# # from sklearn.preprocessing import LabelEncoder

# # from src.preprocess_tfidf import Preprocessor as TFIDFPreprocessor
# # from src.preprocess_bert import PreprocessorBERT
# # from src.train import Trainer
# # from src.predict import Predictor
# # from src.decision import decide_action

# # # =========================
# # # 🔹 LOAD DATA
# # # =========================
# # train = pd.read_csv("data/train.csv")
# # test = pd.read_csv("data/test.csv")

# # train.columns = train.columns.str.strip().str.lower()
# # test.columns = test.columns.str.strip().str.lower()

# # text_col = "journal_text"

# # meta_cols = [
# #     'sleep_hours',
# #     'stress_level',
# #     'energy_level',
# #     'time_of_day',
# #     'ambience_type',
# #     'previous_day_mood',
# #     'face_emotion_hint',
# #     'reflection_quality'
# # ]

# # # =========================
# # # 🔹 LABEL ENCODING
# # # =========================
# # le_state = LabelEncoder()
# # y_state = le_state.fit_transform(train["emotional_state"])

# # le_intensity = LabelEncoder()
# # y_intensity = le_intensity.fit_transform(train["intensity"])

# # # =========================
# # # 🔹 PREPROCESSING
# # # =========================
# # tfidf_pre = TFIDFPreprocessor()
# # bert_pre = PreprocessorBERT()

# # X_text_tfidf, X_meta = tfidf_pre.fit_transform(train, text_col, meta_cols)
# # X_text_bert = bert_pre.fit_transform(train[text_col].tolist())

# # # =========================
# # # 🔹 TRAIN / VAL SPLIT
# # # =========================
# # X_tfidf_tr, X_tfidf_val, X_meta_tr, X_meta_val, X_bert_tr, X_bert_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
# #     X_text_tfidf,
# #     X_meta,
# #     X_text_bert,
# #     y_state,
# #     y_intensity,
# #     test_size=0.2,
# #     random_state=42
# # )

# # # =========================
# # # 🔹 TRAIN
# # # =========================
# # trainer = Trainer()

# # (
# #     model_state_tfidf,
# #     model_intensity_tfidf,
# #     model_state_bert,
# #     model_intensity_bert
# # ) = trainer.train(
# #     X_tfidf_tr,
# #     X_bert_tr,
# #     X_meta_tr,
# #     y_state_tr,
# #     y_int_tr
# # )

# # # =========================
# # # 🔹 VALIDATION (COMPARE)
# # # =========================
# # predictor = Predictor(
# #     model_state_tfidf,
# #     model_intensity_tfidf,
# #     model_state_bert,
# #     model_intensity_bert
# # )

# # pred_val = predictor.predict(
# #     X_tfidf_val,
# #     X_bert_val,
# #     X_meta_val
# # )

# # # TF-IDF validation
# # val_preds_tfidf = pred_val["tfidf"]["state"]
# # val_preds_tfidf_decoded = le_state.inverse_transform(val_preds_tfidf)
# # y_val_decoded = le_state.inverse_transform(y_state_val)

# # print("\n📊 TF-IDF Validation Report:")
# # print(classification_report(y_val_decoded, val_preds_tfidf_decoded))

# # # BERT validation
# # val_preds_bert = pred_val["bert"]["state"]
# # val_preds_bert_decoded = le_state.inverse_transform(val_preds_bert)

# # print("\n📊 BERT Validation Report:")
# # print(classification_report(y_val_decoded, val_preds_bert_decoded))

# # # =========================
# # # 🔹 TEST PREPROCESSING
# # # =========================
# # X_text_tfidf_test, X_meta_test = tfidf_pre.transform(test, text_col, meta_cols)
# # X_text_bert_test = bert_pre.transform(test[text_col].tolist())

# # # =========================
# # # 🔹 PREDICT
# # # =========================
# # pred = predictor.predict(
# #     X_text_tfidf_test,
# #     X_text_bert_test,
# #     X_meta_test
# # )

# # # =========================
# # # 🔹 DECODE
# # # =========================
# # state_tfidf = le_state.inverse_transform(pred["tfidf"]["state"])
# # intensity_tfidf = le_intensity.inverse_transform(pred["tfidf"]["intensity"])

# # state_bert = le_state.inverse_transform(pred["bert"]["state"])
# # intensity_bert = le_intensity.inverse_transform(pred["bert"]["intensity"])

# # # =========================
# # # 🔹 DECISION ENGINE (BOTH)
# # # =========================
# # actions_tfidf, timings_tfidf = [], []
# # actions_bert, timings_bert = [], []

# # for i in range(len(test)):

# #     # TF-IDF decision
# #     a_t, t_t = decide_action(
# #         state_tfidf[i],
# #         intensity_tfidf[i],
# #         test['stress_level'].iloc[i],
# #         test['energy_level'].iloc[i],
# #         test['time_of_day'].iloc[i]
# #     )

# #     # BERT decision
# #     a_b, t_b = decide_action(
# #         state_bert[i],
# #         intensity_bert[i],
# #         test['stress_level'].iloc[i],
# #         test['energy_level'].iloc[i],
# #         test['time_of_day'].iloc[i]
# #     )

# #     actions_tfidf.append(a_t)
# #     timings_tfidf.append(t_t)

# #     actions_bert.append(a_b)
# #     timings_bert.append(t_b)

# # # =========================
# # # 🔹 SAVE OUTPUT (BOTH)
# # # =========================
# # output = pd.DataFrame({
# #     "id": test["id"],

# #     "tfidf_state": state_tfidf,
# #     "tfidf_intensity": intensity_tfidf,
# #     "tfidf_conf": pred["tfidf"]["confidence"],
# #     "tfidf_action": actions_tfidf,

# #     "bert_state": state_bert,
# #     "bert_intensity": intensity_bert,
# #     "bert_conf": pred["bert"]["confidence"],
# #     "bert_action": actions_bert
# # })

# # output.to_csv("outputs/predictions.csv", index=False)


# # print("\n✅ predictions.csv generated!")

# import pandas as pd
# import os
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from src.preprocess_tfidf import Preprocessor as TFIDFPreprocessor
# from src.preprocess_bert import PreprocessorBERT
# from src.train import Trainer
# from src.predict import Predictor
# from src.decision import decide_action


# # =========================
# # 🔹 LOAD DATA
# # =========================
# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")

# train.columns = train.columns.str.strip().str.lower()
# test.columns = test.columns.str.strip().str.lower()

# text_col = "journal_text"

# meta_cols = [
#     'sleep_hours',
#     'stress_level',
#     'energy_level',
#     'time_of_day',
#     'ambience_type',
#     'previous_day_mood',
#     'face_emotion_hint',
#     'reflection_quality'
# ]


# # =========================
# # 🔹 LABEL ENCODING
# # =========================
# le_state = LabelEncoder()
# y_state = le_state.fit_transform(train["emotional_state"])

# le_intensity = LabelEncoder()
# y_intensity = le_intensity.fit_transform(train["intensity"])


# # =========================
# # 🔹 PREPROCESSING
# # =========================
# tfidf_pre = TFIDFPreprocessor()
# bert_pre = PreprocessorBERT()

# X_text_tfidf, X_meta = tfidf_pre.fit_transform(train, text_col, meta_cols)

# # 🔥 BERT embeddings
# X_text_bert = bert_pre.fit_transform(train[text_col].tolist())

# # 🔥 Normalize BERT
# scaler_bert = StandardScaler()
# X_text_bert = scaler_bert.fit_transform(X_text_bert)


# # =========================
# # 🔹 TRAIN / VALIDATION SPLIT
# # =========================
# X_tfidf_tr, X_tfidf_val, X_meta_tr, X_meta_val, X_bert_tr, X_bert_val, y_state_tr, y_state_val, y_int_tr, y_int_val = train_test_split(
#     X_text_tfidf,
#     X_meta,
#     X_text_bert,
#     y_state,
#     y_intensity,
#     test_size=0.2,
#     random_state=42
# )


# # =========================
# # 🔹 TRAIN MODELS
# # =========================
# trainer = Trainer()

# (
#     model_state_tfidf,
#     model_intensity_tfidf,
#     model_state_bert,
#     model_intensity_bert
# ) = trainer.train(
#     X_tfidf_tr,
#     X_bert_tr,
#     X_meta_tr,
#     y_state_tr,
#     y_int_tr
# )


# # =========================
# # 🔹 PREDICTOR
# # =========================
# predictor = Predictor(
#     model_state_tfidf,
#     model_intensity_tfidf,
#     model_state_bert,
#     model_intensity_bert
# )


# # =========================
# # 🔹 VALIDATION
# # =========================
# pred_val = predictor.predict(
#     X_tfidf_val,
#     X_bert_val,
#     X_meta_val
# )

# # TF-IDF
# val_preds_tfidf = pred_val["tfidf"]["state"]
# print("\n📊 TF-IDF Validation Report:")
# print(classification_report(
#     le_state.inverse_transform(y_state_val),
#     le_state.inverse_transform(val_preds_tfidf)
# ))

# # BERT
# val_preds_bert = pred_val["bert"]["state"]
# print("\n📊 BERT Validation Report:")
# print(classification_report(
#     le_state.inverse_transform(y_state_val),
#     le_state.inverse_transform(val_preds_bert)
# ))


# # =========================
# # 🔹 TEST PREPROCESSING
# # =========================
# X_text_tfidf_test, X_meta_test = tfidf_pre.transform(test, text_col, meta_cols)

# X_text_bert_test = bert_pre.transform(test[text_col].tolist())
# X_text_bert_test = scaler_bert.transform(X_text_bert_test)


# # =========================
# # 🔹 PREDICT ON TEST
# # =========================
# pred = predictor.predict(
#     X_text_tfidf_test,
#     X_text_bert_test,
#     X_meta_test
# )


# # =========================
# # 🔹 DECODE RESULTS
# # =========================
# state_tfidf = le_state.inverse_transform(pred["tfidf"]["state"])
# intensity_tfidf = le_intensity.inverse_transform(pred["tfidf"]["intensity"])

# state_bert = le_state.inverse_transform(pred["bert"]["state"])
# intensity_bert = le_intensity.inverse_transform(pred["bert"]["intensity"])


# # =========================
# # 🔹 DECISION ENGINE
# # =========================
# actions_tfidf, timings_tfidf = [], []
# actions_bert, timings_bert = [], []

# for i in range(len(test)):

#     a_t, t_t = decide_action(
#         state_tfidf[i],
#         intensity_tfidf[i],
#         test['stress_level'].iloc[i],
#         test['energy_level'].iloc[i],
#         test['time_of_day'].iloc[i]
#     )

#     a_b, t_b = decide_action(
#         state_bert[i],
#         intensity_bert[i],
#         test['stress_level'].iloc[i],
#         test['energy_level'].iloc[i],
#         test['time_of_day'].iloc[i]
#     )

#     actions_tfidf.append(a_t)
#     timings_tfidf.append(t_t)

#     actions_bert.append(a_b)
#     timings_bert.append(t_b)


# # =========================
# # 🔹 SAVE OUTPUT
# # =========================
# output = pd.DataFrame({
#     "id": test["id"],

#     "tfidf_state": state_tfidf,
#     "tfidf_intensity": intensity_tfidf,
#     "tfidf_conf": pred["tfidf"]["confidence"].round(3),
#     "tfidf_action": actions_tfidf,

#     "bert_state": state_bert,
#     "bert_intensity": intensity_bert,
#     "bert_conf": pred["bert"]["confidence"].round(3),
#     "bert_action": actions_bert
# })

# os.makedirs("outputs", exist_ok=True)
# output.to_csv("outputs/predictions.csv", index=False)

# print("\n✅ predictions.csv generated!")


# # =========================
# # 🔹 SAVE MODELS
# # =========================
# os.makedirs("models", exist_ok=True)

# joblib.dump(model_state_tfidf, "models/state_tfidf.pkl")
# joblib.dump(model_intensity_tfidf, "models/intensity_tfidf.pkl")

# joblib.dump(model_state_bert, "models/state_bert.pkl")
# joblib.dump(model_intensity_bert, "models/intensity_bert.pkl")

# joblib.dump(tfidf_pre, "models/tfidf_pre.pkl")
# joblib.dump(bert_pre, "models/bert_pre.pkl")

# joblib.dump(le_state, "models/le_state.pkl")
# joblib.dump(le_intensity, "models/le_intensity.pkl")

# # 🔥 IMPORTANT FIX
# joblib.dump(scaler_bert, "models/scaler_bert.pkl")

# print("✅ Models saved successfully!")




import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.preprocess_tfidf import Preprocessor as TFIDFPreprocessor
from src.preprocess_bert import PreprocessorBERT
from src.train import Trainer
from src.predict import Predictor
from src.decision import decide_action


# =========================
# 🔹 LOAD DATA
# =========================
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train.columns = train.columns.str.strip().str.lower()
test.columns = test.columns.str.strip().str.lower()

text_col = "journal_text"

meta_cols = [
    'sleep_hours',
    'stress_level',
    'energy_level',
    'time_of_day',
    'ambience_type',
    'previous_day_mood',
    'face_emotion_hint',
    'reflection_quality'
]


# =========================
# 🔹 LABEL ENCODING
# =========================
le_state = LabelEncoder()
y_state = le_state.fit_transform(train["emotional_state"])

le_intensity = LabelEncoder()
y_intensity = le_intensity.fit_transform(train["intensity"])


# =========================
# 🔹 PREPROCESSING
# =========================
tfidf_pre = TFIDFPreprocessor()
bert_pre = PreprocessorBERT()

X_text_tfidf, X_meta = tfidf_pre.fit_transform(train, text_col, meta_cols)

# 🔥 BERT embeddings
X_text_bert = bert_pre.fit_transform(train[text_col].tolist())

# 🔥 Normalize BERT (VERY IMPORTANT)
scaler_bert = StandardScaler()
X_text_bert = scaler_bert.fit_transform(X_text_bert)


# =========================
# 🔹 TRAIN / VALIDATION SPLIT
# =========================
(
    X_tfidf_tr, X_tfidf_val,
    X_meta_tr, X_meta_val,
    X_bert_tr, X_bert_val,
    y_state_tr, y_state_val,
    y_int_tr, y_int_val
) = train_test_split(
    X_text_tfidf,
    X_meta,
    X_text_bert,
    y_state,
    y_intensity,
    test_size=0.2,
    random_state=42
)


# =========================
# 🔹 TRAIN MODELS
# =========================
trainer = Trainer()

(
    model_state_tfidf,
    model_intensity_tfidf,
    model_state_bert,
    model_intensity_bert
) = trainer.train(
    X_tfidf_tr,
    X_bert_tr,
    X_meta_tr,
    y_state_tr,
    y_int_tr
)


# =========================
# 🔹 PREDICTOR (NOW SUPPORTS HYBRID)
# =========================
predictor = Predictor(
    model_state_tfidf,
    model_intensity_tfidf,
    model_state_bert,
    model_intensity_bert
)


# =========================
# 🔹 VALIDATION
# =========================
pred_val = predictor.predict(
    X_tfidf_val,
    X_bert_val,
    X_meta_val
)

# TF-IDF
print("\n📊 TF-IDF Validation:")
print(classification_report(
    le_state.inverse_transform(y_state_val),
    le_state.inverse_transform(pred_val["tfidf"]["state"])
))

# BERT
print("\n📊 BERT Validation:")
print(classification_report(
    le_state.inverse_transform(y_state_val),
    le_state.inverse_transform(pred_val["bert"]["state"])
))


# =========================
# 🔥 HYBRID ENSEMBLE (NEW)
# =========================
def hybrid_prediction(pred_dict, bert_weight=0.7):
    tfidf_probs = pred_dict["tfidf"]["probs"]
    bert_probs = pred_dict["bert"]["probs"]

    final_probs = bert_weight * bert_probs + (1 - bert_weight) * tfidf_probs

    final_state = np.argmax(final_probs, axis=1)
    final_conf = np.max(final_probs, axis=1)

    return final_state, final_conf


# =========================
# 🔹 TEST PREPROCESSING
# =========================
X_text_tfidf_test, X_meta_test = tfidf_pre.transform(test, text_col, meta_cols)

X_text_bert_test = bert_pre.transform(test[text_col].tolist())
X_text_bert_test = scaler_bert.transform(X_text_bert_test)


# =========================
# 🔹 PREDICT
# =========================
pred = predictor.predict(
    X_text_tfidf_test,
    X_text_bert_test,
    X_meta_test
)


# =========================
# 🔥 HYBRID PREDICTION
# =========================
hybrid_state_encoded, hybrid_conf = hybrid_prediction(pred)


# =========================
# 🔹 DECODE
# =========================
state_tfidf = le_state.inverse_transform(pred["tfidf"]["state"])
state_bert = le_state.inverse_transform(pred["bert"]["state"])
state_hybrid = le_state.inverse_transform(hybrid_state_encoded)

intensity_tfidf = le_intensity.inverse_transform(pred["tfidf"]["intensity"])
intensity_bert = le_intensity.inverse_transform(pred["bert"]["intensity"])


# =========================
# 🔹 DECISION ENGINE
# =========================
actions_hybrid, timings_hybrid = [], []

for i in range(len(test)):

    a, t = decide_action(
        state_hybrid[i],
        intensity_bert[i],  # better to use BERT intensity
        test['stress_level'].iloc[i],
        test['energy_level'].iloc[i],
        test['time_of_day'].iloc[i]
    )

    actions_hybrid.append(a)
    timings_hybrid.append(t)


# =========================
# 🔹 SAVE OUTPUT
# =========================
output = pd.DataFrame({
    "id": test["id"],

    "tfidf_state": state_tfidf,
    "tfidf_conf": pred["tfidf"]["confidence"].round(3),

    "bert_state": state_bert,
    "bert_conf": pred["bert"]["confidence"].round(3),

    "hybrid_state": state_hybrid,
    "hybrid_conf": hybrid_conf.round(3),
    "hybrid_action": actions_hybrid
})

os.makedirs("outputs", exist_ok=True)
output.to_csv("outputs/predictions.csv", index=False)

print("\n✅ predictions.csv generated!")


# =========================
# 🔹 SAVE MODELS
# =========================
os.makedirs("models", exist_ok=True)

joblib.dump(model_state_tfidf, "models/state_tfidf.pkl")
joblib.dump(model_intensity_tfidf, "models/intensity_tfidf.pkl")

joblib.dump(model_state_bert, "models/state_bert.pkl")
joblib.dump(model_intensity_bert, "models/intensity_bert.pkl")

joblib.dump(tfidf_pre, "models/tfidf_pre.pkl")
joblib.dump(bert_pre, "models/bert_pre.pkl")

joblib.dump(le_state, "models/le_state.pkl")
joblib.dump(le_intensity, "models/le_intensity.pkl")

joblib.dump(scaler_bert, "models/scaler_bert.pkl")

print("✅ Models saved successfully!")