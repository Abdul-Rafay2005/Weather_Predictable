# backend/src/ml/train_model.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("nyc_training_data.csv")

# Features & targets
X = df[["tmax", "tmin", "precip", "wind", "rh", "doy"]]
targets = ["very_hot", "very_cold", "very_wet", "very_windy", "very_uncomfortable"]

os.makedirs("backend/models", exist_ok=True)

def train_and_save_models(X, df):
    models = {}
    for target in targets:
        print(f"\n🚀 Training model for: {target}")
        y = df[target].fillna(0).astype(int)   # ensure 0/1 integers only

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # model
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            base_score=0.5,   # ✅ ensures valid logistic base score
            use_label_encoder=False
        )

        # fit
        model.fit(X_train, y_train)

        # evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ {target} Accuracy: {acc:.2f}")
        print(classification_report(y_test, preds))

        # save model
        model_path = f"backend/models/{target}_model.joblib"
        joblib.dump(model, model_path)
        print(f"📁 Saved: {model_path}")

        models[target] = model
    return models


if __name__ == "__main__":
    train_and_save_models(X, df)
