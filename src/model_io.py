"""تحميل النموذج المحفوظ مع التأكد أن إصدار scikit-learn المستخدم هو نفسه الذي أُنشئ به النموذج."""
import warnings
from pathlib import Path

import joblib
from sklearn.exceptions import InconsistentVersionWarning


def load_model(path):
    """يحمّل الـ Pipeline المحفوظ، ويرفع RuntimeError واضحًا لو اختلف إصدار scikit-learn.

    الاختلاف بين إصدار التدريب وإصدار التشغيل قد يعطي نتائج خاطئة بدون أي خطأ ظاهر،
    لذلك نوقف التشغيل بدل الاكتفاء بتحذير.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ملف النموذج غير موجود: {path}\nشغّل التدريب أولًا: python -m src.train")

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning as warning:
            raise RuntimeError(
                "إصدار scikit-learn الحالي يختلف عن الإصدار الذي أُنشئ به النموذج المحفوظ.\n"
                f"({warning.original_sklearn_version} عند التدريب، و{warning.current_sklearn_version} الآن)\n"
                "الحل: ثبّت الإصدار المحدد في requirements.txt ثم أعد إنشاء النموذج بـ: python -m src.train"
            ) from warning
