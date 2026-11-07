"""سجل بسيط للمرضى: كل زيارة تُحفظ في SQLite، ويمكن مقارنة الزيارات وعرض تطورها.

الملف يقع في data/patients.db (ويمكن تغييره بالمتغير HEART_DB_PATH).
مهم: استخدم رمزًا للمريض (مثل P001) وليس اسمه الحقيقي.
"""
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "patients.db"

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalch", "exang", "oldpeak", "slope"]
TRACKED = ["probability", "trestbps", "chol", "thalch", "oldpeak"]   # ما تتم مقارنته بين الزيارات
BOOLEAN = ["fbs", "exang"]

PATIENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id  TEXT NOT NULL,
    visit_date  TEXT NOT NULL,
    age         REAL,
    sex         TEXT,
    cp          TEXT,
    trestbps    REAL,
    chol        REAL,
    fbs         INTEGER,
    restecg     TEXT,
    thalch      REAL,
    exang       INTEGER,
    oldpeak     REAL,
    slope       TEXT,
    probability REAL NOT NULL,
    priority    TEXT NOT NULL,
    alerts      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits (patient_id, visit_date, id);
"""


def valid_patient_id(value):
    return isinstance(value, str) and PATIENT_ID_RE.fullmatch(value) is not None


def normalize_date(value=None):
    """يرجع التاريخ بصيغة YYYY-MM-DD (اليوم لو لم يُحدد). يرفع ValueError لو الصيغة غلط."""
    if value is None:
        return date.today().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(value).isoformat()


def _connect(db_path=None):
    path = Path(db_path or os.environ.get("HEART_DB_PATH") or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _clean(value):
    """يحوّل القيمة إلى شيء يقبله SQLite (NaN و None تصبح NULL، والمنطقي يصبح 0/1)."""
    if value is None or value != value:      # value != value تعني NaN
        return None
    if hasattr(value, "item"):               # أنواع numpy
        value = value.item()
    return int(value) if isinstance(value, bool) else value


def add_visit(patient_id, patient, probability, priority, alerts, visit_date=None, db_path=None):
    """يحفظ زيارة جديدة ويرجع رقمها."""
    if not valid_patient_id(patient_id):
        raise ValueError("patient_id يجب أن يكون من حروف إنجليزية وأرقام و - و _ (حتى 64 حرفًا)")
    values = [_clean(patient.get(name)) for name in FEATURES]
    with closing(_connect(db_path)) as conn, conn:
        cursor = conn.execute(
            f"INSERT INTO visits (patient_id, visit_date, {', '.join(FEATURES)}, probability, priority, alerts) "
            f"VALUES (?, ?, {', '.join('?' * len(FEATURES))}, ?, ?, ?)",
            [patient_id, normalize_date(visit_date), *values, float(probability), priority,
             json.dumps(alerts, ensure_ascii=False)],
        )
        return cursor.lastrowid


def list_patients(db_path=None):
    """قائمة المرضى المسجلين مع عدد زياراتهم وتاريخ آخر زيارة."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT patient_id, COUNT(*) AS visits, MAX(visit_date) AS last_visit "
            "FROM visits GROUP BY patient_id ORDER BY last_visit DESC, patient_id"
        ).fetchall()
    return [dict(row) for row in rows]


def get_history(patient_id, db_path=None):
    """زيارات المريض مرتبة من الأقدم إلى الأحدث (DataFrame فارغ لو المريض غير موجود)."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM visits WHERE patient_id = ? ORDER BY visit_date, id", (patient_id,)
        ).fetchall()
    history = pd.DataFrame([dict(row) for row in rows])
    if history.empty:
        return history
    history = history.rename(columns={"id": "visit_id"})
    history.insert(0, "visit_no", range(1, len(history) + 1))
    for name in ["age", "trestbps", "chol", "thalch", "oldpeak", "probability"]:
        history[name] = pd.to_numeric(history[name])       # NULL يصبح NaN بدل None
    history["alerts"] = history["alerts"].map(json.loads)
    for name in BOOLEAN:
        history[name] = history[name].map({1: True, 0: False})
    return history


def history_records(patient_id, db_path=None):
    """نفس get_history لكن كقائمة dicts جاهزة للـ JSON (القيم الناقصة None)."""
    history = get_history(patient_id, db_path)
    if history.empty:
        return []
    history = history.astype(object).where(history.notna(), None)
    return history.to_dict("records")


def compare_visits(history):
    """يقارن آخر زيارتين: ما تغير في الاحتمال والقياسات. يرجع None لو أقل من زيارتين."""
    if len(history) < 2:
        return None
    previous, current = history.iloc[-2], history.iloc[-1]

    changes = []
    for field in TRACKED:
        before, after = previous[field], current[field]
        if pd.isna(before) or pd.isna(after):
            continue
        diff = float(after) - float(before)
        trend = "same" if abs(diff) < 1e-9 else ("up" if diff > 0 else "down")
        changes.append({"field": field, "previous": float(before), "current": float(after),
                        "change": round(diff, 3), "trend": trend})

    def summary(row):
        return {"visit_no": int(row["visit_no"]), "visit_date": row["visit_date"], "priority": row["priority"]}

    return {"previous_visit": summary(previous), "current_visit": summary(current), "changes": changes}
