"""واجهة ويب بسيطة (Streamlit) لتوقع مرض القلب ومتابعة المريض.

التشغيل:  streamlit run app.py   (بعد تدريب النموذج: python -m src.train)
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from src import records
from src.alerts import PRIORITY_ADVICE, PRIORITY_LABELS, build_alerts
from src.model_io import load_model
from src.train import load_data


MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_disease_pipeline.pkl"
DISCLAIMER = "تنبؤ مبدئي لأغراض تعليمية، وليس تشخيصًا طبيًا."

PRIORITY_DISPLAY = {"urgent": st.error, "high": st.warning, "moderate": st.info, "routine": st.success}
ALERT_DISPLAY = {"critical": st.error, "warning": st.warning, "info": st.info}

FIELD_LABELS = {          # ما يظهر في المقارنة ومنحنى التطور
    "probability": "Disease probability",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "thalch": "Max heart rate",
    "oldpeak": "ST depression (oldpeak)",
}


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)                          # يتأكد أيضًا من تطابق إصدار scikit-learn
    patients = load_data()                                  # بيانات المرضى بعد التنظيف (فيها target)
    features = patients.drop(columns=["target"])
    vectors = model.named_steps["prep"].transform(features)  # تمثيل رقمي لكل مريض للمقارنة
    return model, patients, features, vectors


model, patients, features, vectors = load_artifacts()


def optional_number(label, key, low, high, default, step):
    """رقم اختياري: لو اختار المستخدم 'unknown' نرجع NaN ويعوّضه النموذج."""
    value = st.number_input(label, low, high, default, step, key=key)
    if st.checkbox(f"{label}: unknown", key=f"{key}_unknown"):
        return np.nan
    return value


def optional_choice(label, key, options, default):
    choice = st.selectbox(label, ["Unknown"] + options, index=options.index(default) + 1, key=key)
    return np.nan if choice == "Unknown" else choice


def optional_yes_no(label, key, default="No"):
    options = ["Unknown", "No", "Yes"]
    choice = st.selectbox(label, options, index=options.index(default), key=key)
    return {"Unknown": np.nan, "No": False, "Yes": True}[choice]


def show_review(review):
    """يعرض أولوية المراجعة والتنبيهات المبنية على بيانات المريض."""
    PRIORITY_DISPLAY[review["priority"]](
        f"**Review priority: {review['priority'].upper()} ({review['priority_label']})** — {review['advice']}"
    )
    for alert in review["alerts"]:
        ALERT_DISPLAY[alert["level"]](alert["message"])


def format_change(change):
    """يرجع القيمة الحالية والفرق بنص مناسب لـ st.metric."""
    if change["field"] == "probability":
        return f"{change['current']:.1%}", f"{change['change'] * 100:+.1f} pts"
    return f"{change['current']:g}", f"{change['change']:+g}"


st.set_page_config(page_title="Heart Disease Diagnosis System", page_icon="❤️")
st.title("Heart disease risk assessment")
st.write("Enter the patient's data to get a preliminary prediction, patient-specific alerts, and the most similar "
         "past patients. Use a patient ID to keep a record of each visit and follow the patient over time.")

tab_assess, tab_history = st.tabs(["Assessment", "Patient history"])

with tab_assess:
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", 1, 120, 55, key="age")
        sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
        cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
                          index=3, key="cp")
        restecg = optional_choice("Resting ECG", "restecg", ["normal", "st-t abnormality", "lv hypertrophy"], "normal")
        slope = optional_choice("ST slope", "slope", ["upsloping", "flat", "downsloping"], "flat")
    with right:
        trestbps = optional_number("Resting blood pressure (mm Hg)", "trestbps", 50, 260, 140, 1)
        chol = optional_number("Cholesterol (mg/dl)", "chol", 50, 700, 250, 1)
        thalch = optional_number("Max heart rate", "thalch", 40, 250, 120, 1)
        oldpeak = optional_number("ST depression (oldpeak)", "oldpeak", -5.0, 10.0, 2.0, 0.1)
        fbs = optional_yes_no("Fasting blood sugar > 120", "fbs")
        exang = optional_yes_no("Exercise-induced chest pain", "exang", default="Yes")

    id_col, date_col = st.columns(2)
    with id_col:
        patient_id = st.text_input("Patient ID (optional — saves this visit to the patient's record)",
                                   key="patient_id", help="Use a code such as P001, not the patient's real name.").strip()
    with date_col:
        visit_date = st.date_input("Visit date", value=date.today(), max_value=date.today(), key="visit_date")

    if st.button("Analyze Patient"):
        patient_values = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
            "restecg": restecg, "thalch": thalch, "exang": exang, "oldpeak": oldpeak, "slope": slope,
        }
        patient = pd.DataFrame([patient_values], dtype=object)   # object مثل أعمدة fbs و exang أثناء التدريب

        probability = float(model.predict_proba(patient)[0, 1])
        sick = probability >= 0.5

        col1, col2 = st.columns(2)
        with col1:
            if sick:
                st.error("**Result:** Heart disease detected")
            else:
                st.success("**Result:** No heart disease detected")
        with col2:
            st.info(f"**Probability of disease:** {probability:.1%}")

        review = build_alerts(patient_values, probability)
        st.subheader("Alerts and recommendations")
        show_review(review)

        if patient_id and not records.valid_patient_id(patient_id):
            st.error("Patient ID must contain only English letters, digits, '-' or '_' (up to 64 characters), "
                     "so this visit was not saved.")
        elif patient_id:
            records.add_visit(patient_id, patient_values, probability, review["priority"], review["alerts"],
                              visit_date=visit_date)
            st.success(f"Visit saved for patient {patient_id}. Open the 'Patient history' tab to follow progress.")

        similarities = cosine_similarity(model.named_steps["prep"].transform(patient), vectors).flatten()
        top_indices = similarities.argsort()[-3:][::-1]

        st.subheader("Top 3 Similar Past Patients:")
        for i, idx in enumerate(top_indices, 1):
            diagnosis = "Heart disease" if patients["target"].iloc[idx] == 1 else "Healthy"
            with st.expander(f"{i}. Diagnosis: {diagnosis} (Similarity Score: {similarities[idx]:.4f})"):
                st.write(features.iloc[idx].fillna("unknown").astype(str))

        st.caption(DISCLAIMER)

with tab_history:
    st.subheader("Patient history")
    known_patients = records.list_patients()
    if not known_patients:
        st.info("No saved visits yet. Enter a Patient ID in the 'Assessment' tab and analyze a patient to start a record.")
    else:
        selected = st.selectbox("Patient", [p["patient_id"] for p in known_patients], key="history_patient")
        history = records.get_history(selected)

        st.write(f"**{len(history)} visit(s)** — latest on {history['visit_date'].iloc[-1]}")
        table = history[["visit_no", "visit_date", "probability", "priority", "age", "trestbps", "chol",
                         "thalch", "oldpeak", "exang"]].copy()
        table["probability"] = (table["probability"] * 100).round(1)
        st.dataframe(table.rename(columns={"probability": "probability (%)"}), hide_index=True)

        st.markdown("#### Progress over time")
        trend_options = {label: field for field, label in FIELD_LABELS.items()}
        trend_label = st.selectbox("Show trend of", list(trend_options), key="trend_field")
        trend_field = trend_options[trend_label]
        series = history.set_index("visit_no")[trend_field]
        st.line_chart(series * 100 if trend_field == "probability" else series,
                      x_label="Visit number", y_label="Probability (%)" if trend_field == "probability" else trend_label)

        st.markdown("#### Last visit compared with the previous one")
        comparison = records.compare_visits(history)
        if comparison is None:
            st.info("At least two visits are needed to compare.")
        else:
            st.caption(f"Visit {comparison['previous_visit']['visit_no']} ({comparison['previous_visit']['visit_date']}) → "
                       f"Visit {comparison['current_visit']['visit_no']} ({comparison['current_visit']['visit_date']}) · "
                       f"priority: {comparison['previous_visit']['priority']} → {comparison['current_visit']['priority']}")
            columns = st.columns(max(len(comparison["changes"]), 1))
            for column, change in zip(columns, comparison["changes"]):
                value, delta = format_change(change)
                column.metric(FIELD_LABELS[change["field"]], value, delta,
                              delta_color="off" if change["field"] == "thalch" else "inverse")

        with st.expander("Alerts from the latest visit"):
            last_priority = history["priority"].iloc[-1]
            show_review({"priority": last_priority, "priority_label": PRIORITY_LABELS[last_priority],
                         "advice": PRIORITY_ADVICE[last_priority], "alerts": history["alerts"].iloc[-1]})
