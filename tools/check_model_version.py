"""يتأكد أن إصدار scikit-learn متطابق في ثلاثة أماكن: النموذج المحفوظ، والبيئة الحالية، وrequirements.txt.

التشغيل:  python tools/check_model_version.py
يرجع كود خروج 1 لو وُجد اختلاف.
"""
import re
import sys
import warnings
from pathlib import Path

import joblib
import sklearn
from packaging.specifiers import SpecifierSet
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.pkl"
REQUIREMENTS = ROOT / "requirements.txt"


def required_spec():
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*scikit-learn\s*([^#\s]*)", line, re.IGNORECASE)
        if match:
            return match.group(1) or None
    return None


def trained_version():
    """إصدار scikit-learn الذي أُنشئ به النموذج (يُكتشف من تحذير التحميل لو اختلف)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joblib.load(MODEL_PATH)
    for item in caught:
        if issubclass(item.category, InconsistentVersionWarning):
            return item.message.original_sklearn_version
    return sklearn.__version__    # لا تحذير = نفس إصدار البيئة الحالية


def main():
    installed = sklearn.__version__
    spec = required_spec()
    trained = trained_version()

    print(f"النموذج المحفوظ أُنشئ بـ : scikit-learn {trained}")
    print(f"المثبّت في البيئة الآن   : scikit-learn {installed}")
    print(f"المطلوب في requirements : scikit-learn {spec or '(غير محدد)'}")

    problems = []
    if trained != installed:
        problems.append("النموذج المحفوظ أُنشئ بإصدار غير الإصدار المثبّت: أعد إنشاءه بـ python -m src.train")
    if spec and installed not in SpecifierSet(spec):
        problems.append("الإصدار المثبّت لا يطابق requirements.txt: نفّذ pip install -r requirements.txt")
    if spec and trained not in SpecifierSet(spec):
        problems.append("إصدار النموذج لا يطابق requirements.txt: أعد التدريب أو عدّل requirements.txt")

    if problems:
        print("\nيوجد تعارض:")
        for problem in problems:
            print(f" - {problem}")
        sys.exit(1)
    print("\nالإصدارات متطابقة.")


if __name__ == "__main__":
    main()
