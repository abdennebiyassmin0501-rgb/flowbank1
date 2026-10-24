"""
risk.py — المؤشرات المالية: PD وLGD وEAD وECL.

    ECL = PD × LGD × EAD

- PD  (احتمال التعثر): تنبؤ فعلي من نموذج مدرَّب على البيانات الحقيقية.
- LGD (الخسارة عند التعثر): افتراض توضيحي يحدده المستخدم؛ البيانات لا تحتوي مبالغ مستردّة.
- EAD (التعرض عند التعثر):
    الأفراد: محسوب من بيانات حقيقية (الرصيد وسقف البطاقة) + افتراض CCF للجزء غير المستعمل.
    الشركات: مبلغ افتراضي يدخله المستخدم؛ بيانات الشركات نِسَب مالية فقط بلا مبالغ قروض.
"""
import numpy as np

DEFAULT_LGD = 0.45   # مرجع شائع: 45% في منهج Basel II التأسيسي للديون غير المضمونة ذات الأولوية
DEFAULT_CCF = 0.50   # افتراض توضيحي: نصف السقف غير المستعمل قد يُسحب قبل التعثر

# حدود مستويات المخاطر: اختيار إداري توضيحي قابل للتعديل في الواجهة
DEFAULT_BANDS = {"individuals": (0.15, 0.35), "companies": (0.05, 0.15)}
RISK_NAMES = {"low": "منخفض", "mid": "متوسط", "high": "مرتفع"}


def _check_rate(name, v):
    if v is None or not np.isfinite(v) or v < 0 or v > 1:
        raise ValueError(f"{name} يجب أن يكون بين 0% و100%.")


def card_ead(limit_bal, bill_amt, ccf=DEFAULT_CCF):
    """EAD لبطاقة ائتمان = المستعمل حاليًا + CCF × الجزء غير المستعمل من السقف."""
    _check_rate("CCF", ccf)
    limit = np.asarray(limit_bal, dtype=float)
    if np.any(limit <= 0):
        raise ValueError("سقف البطاقة يجب أن يكون أكبر من صفر.")
    drawn = np.maximum(np.asarray(bill_amt, dtype=float), 0.0)
    undrawn = np.maximum(limit - drawn, 0.0)
    ead = drawn + ccf * undrawn
    return (float(ead), float(drawn), float(undrawn)) if np.ndim(ead) == 0 else (ead, drawn, undrawn)


def ecl(pd_, lgd, ead):
    _check_rate("PD", pd_)
    _check_rate("LGD", lgd)
    if ead is None or not np.isfinite(ead) or ead < 0:
        raise ValueError("EAD يجب أن يكون مبلغًا موجبًا.")
    return pd_ * lgd * ead


def portfolio_ecl(pds, lgd, eads):
    pds, eads = np.asarray(pds, dtype=float), np.asarray(eads, dtype=float)
    _check_rate("LGD", lgd)
    if np.any((pds < 0) | (pds > 1)) or np.any(eads < 0):
        raise ValueError("قيم PD أو EAD خارج المجال المسموح.")
    return float(np.sum(pds * lgd * eads))


# جداول القرار من وثيقة «دراسات حالة: اتخاذ القرار الائتماني» (المبالغ بالدينار الجزائري)
DECISION_TABLES = {
    "individuals": {"limits": (3_000, 8_000),
                    "texts": {"accept": "قبول البطاقة", "review": "قبول بحد أقل أو طلب ضمان", "reject": "رفض"}},
    "companies": {"limits": (15_000, 40_000),
                  "texts": {"accept": "قبول القرض", "review": "قبول بمبلغ أقل أو طلب ضمان إضافي", "reject": "رفض القرض"}},
}


def decide(el, group):
    """القرار حسب الخسارة المتوقعة: أقل من الحد الأول قبول، حتى الحد الثاني مراجعة، وفوقه رفض."""
    if el is None or not np.isfinite(el) or el < 0:
        raise ValueError("الخسارة المتوقعة يجب أن تكون مبلغًا موجبًا.")
    low, high = DECISION_TABLES[group]["limits"]
    key = "accept" if el < low else ("review" if el <= high else "reject")
    return key, DECISION_TABLES[group]["texts"][key]


def loan_ead(principal, repaid, accrued_interest=0.0):
    """EAD لقرض بمبلغ محدد = المبلغ المتبقي (الأصل − المسدَّد) + الفوائد المستحقة غير المدفوعة."""
    for name, v in (("مبلغ القرض", principal), ("المبلغ المسدَّد", repaid), ("الفوائد المستحقة", accrued_interest)):
        if v is None or not np.isfinite(v) or v < 0:
            raise ValueError(f"{name} يجب أن يكون مبلغًا موجبًا.")
    if repaid > principal:
        raise ValueError("المبلغ المسدَّد أكبر من مبلغ القرض.")
    return principal - repaid + accrued_interest


def lgd_from_recovery(ead, collateral, haircut, costs, unsecured_recovery):
    """
    LGD = 1 − (المبلغ المسترد الصافي ÷ EAD)
    المسترد = min(EAD, الضمان × (1 − الاقتطاع)) + الجزء غير المغطّى × نسبة استرداده − تكاليف التحصيل
    ويُحصر المسترد بين 0 وEAD، فتبقى LGD بين 0% و100%.
    """
    if ead is None or not np.isfinite(ead) or ead <= 0:
        raise ValueError("EAD يجب أن يكون أكبر من صفر لحساب LGD.")
    for name, v in (("قيمة الضمان", collateral), ("تكاليف التحصيل", costs)):
        if v is None or not np.isfinite(v) or v < 0:
            raise ValueError(f"{name} يجب أن تكون مبلغًا موجبًا.")
    _check_rate("نسبة الاقتطاع", haircut)
    _check_rate("نسبة الاسترداد من الجزء غير المضمون", unsecured_recovery)
    secured = min(ead, collateral * (1 - haircut))
    unsecured = ead - secured
    recovered = min(max(secured + unsecured * unsecured_recovery - costs, 0.0), ead)
    return {"secured": secured, "unsecured": unsecured, "recovered": recovered,
            "loss": ead - recovered, "lgd": 1 - recovered / ead}


def safe_ratio(a, b):
    """نسبة a ÷ b، أو None إذا كان المقام صفرًا أو قيمة ناقصة."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def risk_band(p, bands):
    low, high = bands
    if not 0 <= low < high <= 1:
        raise ValueError("الحد الأول يجب أن يكون أقل من الحد الثاني، وكلاهما بين 0% و100%.")
    return "low" if p < low else ("mid" if p < high else "high")


# ======================================================================= قرار دراسات الحالة
# حدود EL من مستند «دراسات حالة: اتخاذ القرار الائتماني» (بالدينار)، قابلة للتعديل في الواجهة.
CASE_THRESHOLDS = {"individual": (3_000, 8_000), "company": (15_000, 40_000)}
CASE_DECISIONS = {
    "individual": {"accept": "قبول البطاقة", "conditional": "قبول بحد أقل أو طلب ضمان", "reject": "رفض"},
    "company": {"accept": "قبول القرض", "conditional": "قبول بمبلغ أقل أو طلب ضمان إضافي", "reject": "رفض القرض"},
}


def case_decision(el, thresholds):
    low, high = thresholds
    if not 0 <= low < high:
        raise ValueError("الحد الأول لجدول القرار يجب أن يكون أقل من الحد الثاني.")
    if el < low:
        return "accept"
    return "conditional" if el <= high else "reject"
