"""Flask API: توقع مبدئي لمرض القلب.

التشغيل:  python api.py   (بعد تدريب النموذج: python -m src.train)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_disease_pipeline.pkl"
ALERT_THRESHOLD = 0.80   # احتمال الإصابة فوق هذا الرقم يطلق تنبيهًا مبكرًا للطبيب
DISCLAIMER = "تنبؤ مبدئي لأغراض تعليمية، وليس تشخيصًا طبيًا."

NUMERIC = {            # الحد الأدنى والأقصى المنطقي لكل قياس
    "age": (1, 120),
    "trestbps": (50, 260),
    "chol": (50, 700),
    "thalch": (40, 250),
    "oldpeak": (-5, 10),
}
CATEGORICAL = {        # القيم المسموحة لكل معلومة وصفية
    "sex": ["Male", "Female"],
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "slope": ["upsloping", "flat", "downsloping"],
}
BOOLEAN = ["fbs", "exang"]
REQUIRED = ["age", "sex", "cp"]    # الباقي اختياري، والنموذج يعوض الناقص

app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False         # يحافظ على ترتيب الحقول في الرد
model = joblib.load(MODEL_PATH)    # يُحمَّل مرة واحدة، ولا يوجد تدريب هنا


def check_input(data):
    """يرجع قائمة بالأخطاء (فارغة لو البيانات سليمة)."""
    errors = []
    if not isinstance(data, dict):
        return ["يجب إرسال بيانات المريض بصيغة JSON object"]

    for name in REQUIRED:
        if data.get(name) is None:
            errors.append(f"الحقل '{name}' مطلوب")

    for name, (low, high) in NUMERIC.items():
        value = data.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"'{name}' يجب أن يكون رقمًا")
        elif not low <= value <= high:
            errors.append(f"'{name}' يجب أن يكون بين {low} و{high}")

    for name, allowed in CATEGORICAL.items():
        value = data.get(name)
        if value is not None and value not in allowed:
            errors.append(f"'{name}' يجب أن يكون واحدًا من {allowed}")

    for name in BOOLEAN:
        value = data.get(name)
        if value is not None and not isinstance(value, bool):
            errors.append(f"'{name}' يجب أن يكون true أو false")

    return errors


def to_dataframe(data):
    """يحوّل بيانات المريض إلى صف واحد بنفس شكل بيانات التدريب."""
    row = {}
    for name in list(NUMERIC) + list(CATEGORICAL) + BOOLEAN:
        value = data.get(name)
        # القيمة غير المرسلة تصبح NaN (وليس None) حتى يعوضها النموذج مثل بيانات التدريب
        row[name] = np.nan if value is None else value
    return pd.DataFrame([row], dtype=object)   # object مثل أعمدة fbs/exang في النوتبوك


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    errors = check_input(data)
    if errors:
        return jsonify({"errors": errors}), 400

    probability = float(model.predict_proba(to_dataframe(data))[0, 1])
    sick = probability >= 0.5

    return jsonify({
        "prediction": int(sick),
        "result": "Heart disease detected" if sick else "No heart disease detected",
        "probability": round(probability, 3),
        "early_alert": probability >= ALERT_THRESHOLD,
        "disclaimer": DISCLAIMER,
    })


if __name__ == "__main__":
    app.run(debug=False)
