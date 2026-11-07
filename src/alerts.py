"""تنبيهات وتوصيات مبنية على بيانات المريض نفسها، وليس على نسبة التوقع فقط.

القواعد تعليمية وبسيطة (حدود شائعة لضغط الدم والكوليسترول وغيرها)، وهدفها مساعدة
الطبيب على ترتيب أولويات المراجعة. هي ليست تشخيصًا ولا بديلًا عن التقييم الطبي.
"""
import math

URGENT_PROBABILITY = 0.80   # فوق هذا الرقم: أولوية قصوى للمراجعة (التنبيه المبكر)
HIGH_PROBABILITY = 0.50     # فوق هذا الرقم: النموذج يتوقع وجود المرض

CRITICAL, WARNING, INFO = "critical", "warning", "info"

PRIORITY_LABELS = {
    "urgent": "عاجلة",
    "high": "مرتفعة",
    "moderate": "متوسطة",
    "routine": "عادية",
}
PRIORITY_ADVICE = {
    "urgent": "مراجعة الطبيب فورًا وبأعلى أولوية.",
    "high": "حجز مراجعة قريبة مع الطبيب.",
    "moderate": "مراجعة ضمن المتابعة الدورية مع الانتباه للتنبيهات.",
    "routine": "متابعة روتينية.",
}


def _is_missing(value):
    """True لو القيمة غير موجودة (None أو NaN)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _number(patient, name):
    """القيمة الرقمية، أو None لو غير موجودة أو NaN."""
    value = patient.get(name)
    if _is_missing(value) or isinstance(value, bool):
        return None
    return value


def _flag(patient, name):
    """True/False للحقول المنطقية، أو None لو غير معروفة."""
    value = patient.get(name)
    return None if _is_missing(value) else bool(value)


def _text(patient, name):
    value = patient.get(name)
    return value if isinstance(value, str) else None


def _alert(code, level, message):
    return {"code": code, "level": level, "message": message}


def _measurement_alerts(patient):
    """تنبيهات مبنية على القياسات والمؤشرات القلبية للمريض."""
    alerts = []

    bp = _number(patient, "trestbps")
    if bp is not None:
        if bp >= 180:
            alerts.append(_alert("bp_very_high", CRITICAL,
                                 f"ضغط الدم مرتفع جدًا ({bp:g} mm Hg): يلزم تقييم طبي سريع ومتابعة ضغط الدم."))
        elif bp >= 140:
            alerts.append(_alert("bp_high", WARNING,
                                 f"ضغط الدم مرتفع ({bp:g} mm Hg): يلزم متابعة ضغط الدم بانتظام."))
        elif bp >= 130:
            alerts.append(_alert("bp_borderline", INFO,
                                 f"ضغط الدم أعلى قليلًا من المعدل ({bp:g} mm Hg): يُفضّل متابعته."))

    chol = _number(patient, "chol")
    if chol is not None:
        if chol >= 240:
            alerts.append(_alert("chol_high", WARNING,
                                 f"الكوليسترول مرتفع ({chol:g} mg/dl): يُنصح بمراجعة مستوى الدهون."))
        elif chol >= 200:
            alerts.append(_alert("chol_borderline", INFO,
                                 f"الكوليسترول عند الحد الأعلى ({chol:g} mg/dl): يُفضّل متابعة مستوى الدهون."))

    if _flag(patient, "fbs"):
        alerts.append(_alert("fbs_high", INFO,
                             "سكر الصائم أعلى من 120 mg/dl: يُنصح بفحص مستوى السكر."))

    if _flag(patient, "exang"):
        alerts.append(_alert("exang", WARNING,
                             "ألم صدر مصاحب للمجهود (exang): ينبّه الطبيب لإجراء تقييم إضافي مثل اختبار الجهد."))

    oldpeak = _number(patient, "oldpeak")
    if oldpeak is not None:
        if oldpeak >= 2:
            alerts.append(_alert("oldpeak_high", WARNING,
                                 f"هبوط واضح في مقطع ST (oldpeak = {oldpeak:g}): مؤشر قلبي يستدعي تقييمًا إضافيًا."))
        elif oldpeak >= 1:
            alerts.append(_alert("oldpeak_borderline", INFO,
                                 f"هبوط بسيط في مقطع ST (oldpeak = {oldpeak:g}): يُفضّل متابعته."))

    if _text(patient, "slope") == "downsloping":
        alerts.append(_alert("slope_down", WARNING,
                             "ميل مقطع ST هابط (downsloping): مؤشر قلبي يستدعي مراجعة الطبيب."))

    if _text(patient, "restecg") in ("st-t abnormality", "lv hypertrophy"):
        alerts.append(_alert("ecg_abnormal", WARNING,
                             "تخطيط القلب أثناء الراحة غير طبيعي: يُنصح بمراجعته مع الطبيب."))

    if _text(patient, "cp") == "asymptomatic":
        alerts.append(_alert("cp_asymptomatic", INFO,
                             "نوع ألم الصدر asymptomatic لا يعني غياب المرض بالضرورة، فيُنصح بمراجعة بقية المؤشرات."))

    age, max_hr = _number(patient, "age"), _number(patient, "thalch")
    if age is not None and max_hr is not None and max_hr < 0.7 * (220 - age):
        alerts.append(_alert("max_hr_low", INFO,
                             f"أقصى معدل لضربات القلب ({max_hr:g}) منخفض بالنسبة للعمر: يُفضّل ذكره للطبيب."))

    missing = [label for name, label in (("trestbps", "ضغط الدم"), ("chol", "الكوليسترول"),
                                         ("exang", "ألم الصدر مع المجهود"))
               if _is_missing(patient.get(name))]
    if missing:
        alerts.append(_alert("missing_data", INFO,
                             "قياسات غير متوفرة: " + "، ".join(missing) + ". التنبيهات المرتبطة بها غير متاحة."))

    return alerts


def build_alerts(patient, probability):
    """يرجع أولوية المراجعة والتنبيهات.

    patient: dict بنفس أسماء أعمدة التدريب (القيم الناقصة None أو NaN).
    probability: احتمال الإصابة من النموذج (0 إلى 1).
    """
    alerts = _measurement_alerts(patient)

    if probability >= URGENT_PROBABILITY:
        alerts.insert(0, _alert("risk_very_high", CRITICAL,
                                f"احتمال الخطر مرتفع جدًا ({probability:.0%}): أولوية قصوى للمراجعة."))
    elif probability >= HIGH_PROBABILITY:
        alerts.insert(0, _alert("risk_high", WARNING,
                                f"احتمال الخطر مرتفع ({probability:.0%}): يلزم مراجعة قريبة."))

    levels = [a["level"] for a in alerts]
    if CRITICAL in levels:
        priority = "urgent"
    elif probability >= HIGH_PROBABILITY or levels.count(WARNING) >= 3:
        priority = "high"
    elif WARNING in levels:
        priority = "moderate"
    else:
        priority = "routine"

    return {
        "priority": priority,
        "priority_label": PRIORITY_LABELS[priority],
        "advice": PRIORITY_ADVICE[priority],
        "alerts": alerts,
    }
