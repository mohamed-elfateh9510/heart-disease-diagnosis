"""تدريب النموذج النهائي (SVM) وحفظه.

التشغيل من مجلد المشروع:
    python -m src.train

الخطوات هنا هي نفسها خطوات النوتبوك (notebooks/medical_diagnosis.ipynb).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.pkl"

NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope"]


def load_data():
    df = pd.read_csv(DATA_PATH)

    # 1) العمود الهدف: 1 = مصاب، 0 = سليم
    df["target"] = (df["num"] > 0).astype(int)

    # 2) حذف الأعمدة غير المفيدة أو الناقصة جدًا
    df = df.drop(columns=["id", "dataset", "num", "ca", "thal"])

    # 3) الصفر في الكوليسترول والضغط قيمة غير حقيقية، نعتبرها مفقودة
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    # ملاحظة: fbs و exang تبقى True/False/NaN كما في النوتبوك، والـ API يرسلها بنفس الشكل

    return df


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC),
        ("cat", categorical_pipeline, CATEGORICAL),
    ])
    # probability=True ليعطينا نسبة احتمال الإصابة، وهي التي تعرضها الـ API
    model = SVC(kernel="rbf", C=1, gamma="scale", probability=True, random_state=42)
    return Pipeline([("prep", preprocessor), ("model", model)])


def main():
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred, target_names=["Healthy", "Disease"]))
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("ROC-AUC :", round(roc_auc_score(y_test, proba), 4))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
