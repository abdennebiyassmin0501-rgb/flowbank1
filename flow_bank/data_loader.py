"""
data_loader.py — قراءة البيانات الحقيقية والتحقق منها، وأسماء المتغيرات بالعربية.

البيانات:
- الأفراد: Default of Credit Card Clients (UCI #350) — 30 000 عميل بطاقة ائتمان في تايوان، أبريل–سبتمبر 2005.
- الشركات: Taiwanese Bankruptcy Prediction (UCI #572) — 6 819 شركة تايوانية (1999–2009)،
  95 مؤشرًا ماليًا وحالة الإفلاس حسب قواعد بورصة تايوان.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CARD_CSV = DATA_DIR / "credit_card_clients.csv"
COMPANY_CSV = DATA_DIR / "taiwan_companies.csv"
POLISH_CSV = DATA_DIR / "polish_companies_5year.csv"


class DataError(Exception):
    """خطأ في البيانات برسالة عربية مفهومة."""


# ======================================================================= الأفراد
CARD_TARGET = "DEFAULT"
CARD_ID = "ID"
PAY_COLS = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CARD_REQUIRED = [CARD_ID, "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + PAY_COLS + BILL_COLS + PAYAMT_COLS + [CARD_TARGET]
# لا نستعمل الجنس ولا الحالة الاجتماعية في التنبؤ (انظري README: العدالة وعدم التمييز).
CARD_EXCLUDED = ["SEX", "MARRIAGE"]

MONTHS = ["سبتمبر", "أغسطس", "يوليو", "يونيو", "مايو", "أبريل"]
CARD_LABELS = {
    "ID": "رقم العميل",
    "LIMIT_BAL": "سقف البطاقة (DA)",
    "SEX": "الجنس",
    "EDUCATION": "المستوى التعليمي",
    "MARRIAGE": "الحالة الاجتماعية",
    "AGE": "العمر",
    "UTILIZATION": "نسبة استعمال السقف",
    "DEFAULT": "تعثّر في الشهر التالي",
}
for i, c in enumerate(PAY_COLS):
    CARD_LABELS[c] = f"حالة السداد — {MONTHS[i]}"
for i, c in enumerate(BILL_COLS):
    CARD_LABELS[c] = f"مبلغ الكشف — {MONTHS[i]} (DA)"
for i, c in enumerate(PAYAMT_COLS):
    CARD_LABELS[c] = f"المبلغ المسدَّد — {MONTHS[i]} (DA)"

EDUCATION_NAMES = {1: "دراسات عليا", 2: "جامعي", 3: "ثانوي", 4: "أخرى"}


def pay_status_text(v):
    """الرموز −2 و0 موجودة في البيانات لكنها غير موثّقة في وصف UCI؛ نذكر تفسيرها الشائع."""
    v = int(v)
    if v == -2:
        return "−2: لا استعمال (تفسير شائع)"
    if v == -1:
        return "−1: سدّد في الموعد"
    if v == 0:
        return "0: ائتمان متجدد (تفسير شائع)"
    return f"متأخر {v} شهر" if v < 9 else "متأخر 9 أشهر أو أكثر"


def load_individuals(path=CARD_CSV):
    path = Path(path)
    if not path.exists():
        raise DataError(f"ملف بيانات الأفراد غير موجود: data/{path.name}. "
                        "شغّلي prepare_data.py على ملف UCI الأصلي (انظري README).")
    df = pd.read_csv(path)
    missing = [c for c in CARD_REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"ملف الأفراد ناقص الأعمدة: {', '.join(missing)}")
    if df[CARD_REQUIRED].isna().any().any():
        raise DataError("ملف الأفراد يحتوي قيمًا فارغة غير متوقعة.")
    if not set(df[CARD_TARGET].unique()) <= {0, 1}:
        raise DataError("عمود التعثر يجب أن يحتوي 0 و1 فقط.")
    df = df.copy()
    # القيم 0 و5 و6 في التعليم غير موثّقة في وصف UCI؛ نضمّها إلى «أخرى» (4).
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].isin([1, 2, 3, 4]), 4)
    df["UTILIZATION"] = utilization(df["BILL_AMT1"], df["LIMIT_BAL"])
    return df


def utilization(bill, limit):
    """نسبة استعمال السقف = مبلغ آخر كشف ÷ سقف البطاقة (محصورة بين −1 و3 لتجنّب القيم الشاذة)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.asarray(bill, dtype=float) / np.asarray(limit, dtype=float)
    u = np.where(np.isfinite(u), u, 0.0)
    return np.clip(u, -1.0, 3.0)


# ======================================================================= الشركات
COMPANY_TARGET = "BANKRUPT"
COMPANY_ID = "COMPANY_ID"
# "Net Income Flag" ثابت (قيمته 1 لكل الشركات) فلا يحمل أي معلومة؛ نحذفه.
COMPANY_DROPPED = ["Net Income Flag"]

# الأسماء الإنجليزية كما في ملف UCI (بعد حذف المسافة في أولها) مع ترجمة عربية مختصرة.
TW_FEATURES = [
    ("ROA(C) before interest and depreciation before interest", "العائد على الأصول (C) قبل الفوائد والاهتلاك"),
    ("ROA(A) before interest and % after tax", "العائد على الأصول (A) قبل الفوائد وبعد الضريبة"),
    ("ROA(B) before interest and depreciation after tax", "العائد على الأصول (B) قبل الفوائد والاهتلاك وبعد الضريبة"),
    ("Operating Gross Margin", "هامش الربح الإجمالي التشغيلي"),
    ("Realized Sales Gross Margin", "هامش الربح الإجمالي المحقق من المبيعات"),
    ("Operating Profit Rate", "معدل الربح التشغيلي"),
    ("Pre-tax net Interest Rate", "صافي معدل الفائدة قبل الضريبة"),
    ("After-tax net Interest Rate", "صافي معدل الفائدة بعد الضريبة"),
    ("Non-industry income and expenditure/revenue", "الإيرادات والمصاريف غير التشغيلية ÷ الإيرادات"),
    ("Continuous interest rate (after tax)", "معدل الفائدة المستمر (بعد الضريبة)"),
    ("Operating Expense Rate", "معدل المصاريف التشغيلية"),
    ("Research and development expense rate", "معدل مصاريف البحث والتطوير"),
    ("Cash flow rate", "معدل التدفق النقدي"),
    ("Interest-bearing debt interest rate", "معدل فائدة الديون المحمّلة بفوائد"),
    ("Tax rate (A)", "معدل الضريبة"),
    ("Net Value Per Share (B)", "صافي القيمة للسهم (B)"),
    ("Net Value Per Share (A)", "صافي القيمة للسهم (A)"),
    ("Net Value Per Share (C)", "صافي القيمة للسهم (C)"),
    ("Persistent EPS in the Last Four Seasons", "ربحية السهم المستمرة في آخر أربعة فصول"),
    ("Cash Flow Per Share", "التدفق النقدي للسهم"),
    ("Revenue Per Share (Yuan ¥)", "الإيرادات للسهم"),
    ("Operating Profit Per Share (Yuan ¥)", "الربح التشغيلي للسهم"),
    ("Per Share Net profit before tax (Yuan ¥)", "صافي الربح قبل الضريبة للسهم"),
    ("Realized Sales Gross Profit Growth Rate", "نمو الربح الإجمالي المحقق من المبيعات"),
    ("Operating Profit Growth Rate", "نمو الربح التشغيلي"),
    ("After-tax Net Profit Growth Rate", "نمو صافي الربح بعد الضريبة"),
    ("Regular Net Profit Growth Rate", "نمو صافي الربح الاعتيادي"),
    ("Continuous Net Profit Growth Rate", "نمو صافي الربح المستمر"),
    ("Total Asset Growth Rate", "نمو إجمالي الأصول"),
    ("Net Value Growth Rate", "نمو صافي القيمة"),
    ("Total Asset Return Growth Rate Ratio", "نمو العائد على إجمالي الأصول"),
    ("Cash Reinvestment %", "نسبة إعادة استثمار النقد"),
    ("Current Ratio", "نسبة السيولة الجارية"),
    ("Quick Ratio", "نسبة السيولة السريعة"),
    ("Interest Expense Ratio", "نسبة مصاريف الفوائد"),
    ("Total debt/Total net worth", "إجمالي الديون ÷ صافي حقوق الملكية"),
    ("Debt ratio %", "نسبة المديونية"),
    ("Net worth/Assets", "صافي حقوق الملكية ÷ الأصول"),
    ("Long-term fund suitability ratio (A)", "نسبة ملاءمة الأموال طويلة الأجل"),
    ("Borrowing dependency", "الاعتماد على الاقتراض"),
    ("Contingent liabilities/Net worth", "الالتزامات المحتملة ÷ صافي حقوق الملكية"),
    ("Operating profit/Paid-in capital", "الربح التشغيلي ÷ رأس المال المدفوع"),
    ("Net profit before tax/Paid-in capital", "صافي الربح قبل الضريبة ÷ رأس المال المدفوع"),
    ("Inventory and accounts receivable/Net value", "المخزون والذمم المدينة ÷ صافي القيمة"),
    ("Total Asset Turnover", "دوران إجمالي الأصول"),
    ("Accounts Receivable Turnover", "دوران الذمم المدينة"),
    ("Average Collection Days", "متوسط أيام التحصيل"),
    ("Inventory Turnover Rate (times)", "معدل دوران المخزون"),
    ("Fixed Assets Turnover Frequency", "دوران الأصول الثابتة"),
    ("Net Worth Turnover Rate (times)", "دوران صافي حقوق الملكية"),
    ("Revenue per person", "الإيرادات لكل موظف"),
    ("Operating profit per person", "الربح التشغيلي لكل موظف"),
    ("Allocation rate per person", "معدل التخصيص لكل موظف"),
    ("Working Capital to Total Assets", "رأس المال العامل ÷ إجمالي الأصول"),
    ("Quick Assets/Total Assets", "الأصول السريعة ÷ إجمالي الأصول"),
    ("Current Assets/Total Assets", "الأصول المتداولة ÷ إجمالي الأصول"),
    ("Cash/Total Assets", "النقد ÷ إجمالي الأصول"),
    ("Quick Assets/Current Liability", "الأصول السريعة ÷ الالتزامات الجارية"),
    ("Cash/Current Liability", "النقد ÷ الالتزامات الجارية"),
    ("Current Liability to Assets", "الالتزامات الجارية ÷ الأصول"),
    ("Operating Funds to Liability", "الأموال التشغيلية ÷ الالتزامات"),
    ("Inventory/Working Capital", "المخزون ÷ رأس المال العامل"),
    ("Inventory/Current Liability", "المخزون ÷ الالتزامات الجارية"),
    ("Current Liabilities/Liability", "الالتزامات الجارية ÷ إجمالي الالتزامات"),
    ("Working Capital/Equity", "رأس المال العامل ÷ حقوق الملكية"),
    ("Current Liabilities/Equity", "الالتزامات الجارية ÷ حقوق الملكية"),
    ("Long-term Liability to Current Assets", "الالتزامات طويلة الأجل ÷ الأصول المتداولة"),
    ("Retained Earnings to Total Assets", "الأرباح المحتجزة ÷ إجمالي الأصول"),
    ("Total income/Total expense", "إجمالي الدخل ÷ إجمالي المصاريف"),
    ("Total expense/Assets", "إجمالي المصاريف ÷ الأصول"),
    ("Current Asset Turnover Rate", "دوران الأصول المتداولة"),
    ("Quick Asset Turnover Rate", "دوران الأصول السريعة"),
    ("Working capitcal Turnover Rate", "دوران رأس المال العامل"),
    ("Cash Turnover Rate", "دوران النقد"),
    ("Cash Flow to Sales", "التدفق النقدي ÷ المبيعات"),
    ("Fixed Assets to Assets", "الأصول الثابتة ÷ الأصول"),
    ("Current Liability to Liability", "الالتزامات الجارية ÷ الالتزامات"),
    ("Current Liability to Equity", "الالتزامات الجارية ÷ حقوق الملكية"),
    ("Equity to Long-term Liability", "حقوق الملكية ÷ الالتزامات طويلة الأجل"),
    ("Cash Flow to Total Assets", "التدفق النقدي ÷ إجمالي الأصول"),
    ("Cash Flow to Liability", "التدفق النقدي ÷ الالتزامات"),
    ("CFO to Assets", "التدفق النقدي التشغيلي ÷ الأصول"),
    ("Cash Flow to Equity", "التدفق النقدي ÷ حقوق الملكية"),
    ("Current Liability to Current Assets", "الالتزامات الجارية ÷ الأصول المتداولة"),
    ("Liability-Assets Flag", "مؤشر: الالتزامات أكبر من الأصول (1 = نعم)"),
    ("Net Income to Total Assets", "صافي الدخل ÷ إجمالي الأصول"),
    ("Total assets to GNP price", "إجمالي الأصول ÷ الناتج القومي"),
    ("No-credit Interval", "فترة الاكتفاء دون ائتمان"),
    ("Gross Profit to Sales", "الربح الإجمالي ÷ المبيعات"),
    ("Net Income to Stockholder's Equity", "صافي الدخل ÷ حقوق المساهمين"),
    ("Liability to Equity", "الالتزامات ÷ حقوق الملكية"),
    ("Degree of Financial Leverage (DFL)", "درجة الرافعة المالية"),
    ("Interest Coverage Ratio (Interest expense to EBIT)", "نسبة تغطية الفوائد"),
    ("Equity to Liability", "حقوق الملكية ÷ الالتزامات"),
]
ATTRS = [en for en, _ in TW_FEATURES]
COMPANY_LABELS = {en: ar for en, ar in TW_FEATURES}
COMPANY_LABELS.update({COMPANY_ID: "رقم الشركة", COMPANY_TARGET: "أفلست"})


def load_companies(path=COMPANY_CSV):
    path = Path(path)
    if not path.exists():
        raise DataError(f"ملف بيانات الشركات غير موجود: data/{path.name}. "
                        "شغّلي prepare_data.py على ملف UCI الأصلي (انظري README).")
    df = pd.read_csv(path)
    missing = [c for c in [COMPANY_ID] + ATTRS + [COMPANY_TARGET] if c not in df.columns]
    if missing:
        raise DataError(f"ملف الشركات ناقص الأعمدة: {', '.join(missing[:5])}")
    if not set(df[COMPANY_TARGET].unique()) <= {0, 1}:
        raise DataError("عمود الإفلاس يجب أن يحتوي 0 و1 فقط.")
    df = df.copy()
    df[ATTRS] = df[ATTRS].replace([np.inf, -np.inf], np.nan)
    return df


# ======================================================================= الشركات البولندية (لدراسات الحالة)
# Polish Companies Bankruptcy (UCI #365)، ملف السنة الخامسة: نسب مالية عادية (غير مُقيَّسة) وحالة الإفلاس بعد سنة.
# نستعمل فقط النسب التي يمكن حسابها من ميزانية وجدول حسابات بسيطين (مثل دراسة الحالة)،
# حتى يتدرب النموذج على المعلومات نفسها التي سندخلها له.
POLISH_ATTR_AR = {
    "Attr1": "صافي الربح ÷ إجمالي الأصول",
    "Attr2": "إجمالي الالتزامات ÷ إجمالي الأصول",
    "Attr3": "رأس المال العامل ÷ إجمالي الأصول",
    "Attr4": "الأصول المتداولة ÷ الالتزامات قصيرة الأجل",
    "Attr7": "الربح قبل الفوائد والضرائب (EBIT) ÷ إجمالي الأصول",
    "Attr8": "حقوق الملكية ÷ إجمالي الالتزامات",
    "Attr10": "حقوق الملكية ÷ إجمالي الأصول",
    "Attr17": "إجمالي الأصول ÷ إجمالي الالتزامات",
    "Attr22": "ربح النشاط التشغيلي ÷ إجمالي الأصول",
    "Attr27": "ربح النشاط التشغيلي ÷ المصاريف المالية",
    "Attr28": "رأس المال العامل ÷ الأصول الثابتة",
    "Attr38": "رأس المال الدائم ÷ إجمالي الأصول",
    "Attr50": "الأصول المتداولة ÷ إجمالي الالتزامات",
    "Attr51": "الالتزامات قصيرة الأجل ÷ إجمالي الأصول",
    "Attr53": "حقوق الملكية ÷ الأصول الثابتة",
    "Attr54": "رأس المال الدائم ÷ الأصول الثابتة",
    "Attr59": "الالتزامات طويلة الأجل ÷ حقوق الملكية",
}
POLISH_ATTRS = list(POLISH_ATTR_AR)
POLISH_LABELS = {k: f"X{k[4:]}: {v}" for k, v in POLISH_ATTR_AR.items()}


def load_polish(path=POLISH_CSV):
    path = Path(path)
    if not path.exists():
        raise DataError(f"ملف بيانات الشركات البولندية غير موجود: data/{path.name}. "
                        "شغّلي prepare_data.py مع ملف UCI الأصلي (انظري README).")
    df = pd.read_csv(path)
    missing = [c for c in [COMPANY_ID] + POLISH_ATTRS + [COMPANY_TARGET] if c not in df.columns]
    if missing:
        raise DataError(f"ملف الشركات البولندية ناقص الأعمدة: {', '.join(missing[:5])}")
    df = df.copy()
    df[POLISH_ATTRS] = df[POLISH_ATTRS].replace([np.inf, -np.inf], np.nan)
    return df


def polish_ratios(total_assets, total_liabilities, current_assets, current_liabilities, net_profit, ebit, interest):
    """يحسب النسب البولندية من أرقام القوائم المالية (بأي عملة: النسب لا وحدة لها).
    افتراضات مذكورة في الواجهة: حقوق الملكية = الأصول − الالتزامات؛ الأصول الثابتة = الأصول − الأصول المتداولة؛
    الالتزامات طويلة الأجل = الالتزامات − الالتزامات المتداولة؛ ربح النشاط التشغيلي ≈ EBIT؛ المصاريف المالية ≈ الفوائد."""
    ta, tl, ca, cl = map(float, (total_assets, total_liabilities, current_assets, current_liabilities))
    if ta <= 0:
        raise ValueError("إجمالي الأصول يجب أن يكون أكبر من صفر.")
    if min(tl, ca, cl) < 0 or ca > ta or cl > tl:
        raise ValueError("تحققي من الأرقام: الأصول المتداولة ≤ إجمالي الأصول، والالتزامات المتداولة ≤ إجمالي الالتزامات، ولا قيم سالبة.")
    eq, fa, ltl, wc = ta - tl, ta - ca, tl - cl, ca - cl
    nan = float("nan")
    div = lambda a, b: a / b if b not in (0, 0.0) else nan  # noqa: E731
    return {
        "Attr1": div(net_profit, ta), "Attr2": div(tl, ta), "Attr3": div(wc, ta), "Attr4": div(ca, cl),
        "Attr7": div(ebit, ta), "Attr8": div(eq, tl), "Attr10": div(eq, ta), "Attr17": div(ta, tl),
        "Attr22": div(ebit, ta), "Attr27": div(ebit, interest), "Attr28": div(wc, fa), "Attr38": div(eq + ltl, ta),
        "Attr50": div(ca, tl), "Attr51": div(cl, ta), "Attr53": div(eq, fa), "Attr54": div(eq + ltl, fa),
        "Attr59": div(ltl, eq),
    }


def label(col):
    return CARD_LABELS.get(col) or COMPANY_LABELS.get(col) or POLISH_LABELS.get(col) or col
