"""
prepare_data.py — تحويل ملفات UCI الأصلية إلى ملفات CSV نظيفة يقرؤها التطبيق.

لماذا هذا الملف؟
- ملف الأفراد الأصلي بصيغة Excel القديمة (.xls) وفيه سطران للعناوين.
- ملف الشركات (Taiwanese Bankruptcy Prediction) بصيغة CSV وأسماء أعمدته تبدأ بمسافة.
نحوّلهما مرة واحدة إلى CSV داخل مجلد data/ حتى يكون التطبيق أسرع ولا يحتاج مكتبات إضافية.

الاستعمال (مرة واحدة فقط، والملفات الناتجة موجودة أصلًا في المشروع):
    python prepare_data.py "default of credit card clients.xls" "data.csv" "polish+companies+bankruptcy+data.zip"
(يقبل أيضًا ملفات zip كما تُنزَّل من UCI. الملف الثالث، البولندي، اختياري ويُستعمل لدراسات الحالة.)

لا يولّد هذا الملف أي بيانات. إن لم يجد الملفات الأصلية يتوقف برسالة واضحة.
"""
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CARD_CSV = DATA_DIR / "credit_card_clients.csv"
COMPANY_CSV = DATA_DIR / "taiwan_companies.csv"
POLISH_CSV = DATA_DIR / "polish_companies_5year.csv"
POLISH_EXPECTED_ROWS, POLISH_EXPECTED_BANKRUPT = 5_910, 410
MANIFEST = DATA_DIR / "manifest.json"

CARD_EXPECTED_ROWS = 30_000
COMPANY_EXPECTED_ROWS = 6_819
COMPANY_EXPECTED_BANKRUPT = 220


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_zip_members(zf, prefix=""):
    """يمرّ على كل ملفات الأرشيف، بما فيها الأرشيفات المضغوطة داخله."""
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        data = zf.read(name)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                yield from iter_zip_members(inner, prefix + name + "/")
        else:
            yield prefix + name, data


def find_member(path, suffix):
    """يعيد محتوى الملف: مباشرة إن لم يكن zip، أو من داخل الأرشيف."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        if not path.name.lower().endswith(suffix.lower()):
            sys.exit(f"توقعتُ ملفًا ينتهي بـ {suffix}: {path.name}")
        return path.name, path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        for name, data in iter_zip_members(zf):
            if name.lower().endswith(suffix.lower()):
                return name, data
    raise FileNotFoundError(f"لم أجد ملفًا ينتهي بـ {suffix} داخل {path}")


# ------------------------------------------------------------------ الأفراد
def read_xls(data):
    """نقرأ .xls بمكتبة xlrd إن وُجدت، وإلا نحوّله بـ LibreOffice."""
    try:
        return pd.read_excel(io.BytesIO(data), header=1)
    except ImportError:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            sys.exit("لقراءة ملف .xls ثبّتي المكتبة xlrd:  pip install xlrd")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "cards.xls"
            src.write_bytes(data)
            subprocess.run([soffice, "--headless", "--convert-to", "csv", "--outdir", tmp, str(src)],
                           check=True, capture_output=True, timeout=300)
            return pd.read_csv(Path(tmp) / "cards.csv", header=1)


def prepare_cards(zip_path):
    name, data = find_member(zip_path, ".xls")
    df = read_xls(data)
    df = df.rename(columns={"default payment next month": "DEFAULT"})
    expected = (["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0"]
                + [f"PAY_{i}" for i in range(2, 7)] + [f"BILL_AMT{i}" for i in range(1, 7)]
                + [f"PAY_AMT{i}" for i in range(1, 7)] + ["DEFAULT"])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        sys.exit(f"أعمدة غير موجودة في ملف الأفراد: {missing}")
    df = df[expected].apply(pd.to_numeric, errors="raise").astype(np.int64)
    if len(df) != CARD_EXPECTED_ROWS:
        sys.exit(f"عدد سجلات الأفراد {len(df)} بدل {CARD_EXPECTED_ROWS}. الملف ليس الملف الأصلي.")
    if set(df["DEFAULT"].unique()) != {0, 1}:
        sys.exit("عمود التعثر في ملف الأفراد يجب أن يحتوي 0 و1 فقط.")
    return df, name


# ------------------------------------------------------------------ الشركات
def prepare_companies(path):
    name, data = find_member(path, ".csv")
    df = pd.read_csv(io.BytesIO(data))
    df.columns = [c.strip() for c in df.columns]
    if "Bankrupt?" not in df.columns:
        sys.exit("عمود Bankrupt? غير موجود. هل هذا ملف Taiwanese Bankruptcy Prediction؟")
    if len(df) != COMPANY_EXPECTED_ROWS or int(df["Bankrupt?"].sum()) != COMPANY_EXPECTED_BANKRUPT:
        sys.exit(f"الملف فيه {len(df)} شركة و{int(df['Bankrupt?'].sum())} إفلاس، "
                 f"والمتوقع {COMPANY_EXPECTED_ROWS} و{COMPANY_EXPECTED_BANKRUPT}. الملف ليس الملف الأصلي.")
    out = df.drop(columns=["Bankrupt?"]).apply(pd.to_numeric, errors="raise")
    out.insert(0, "COMPANY_ID", [f"TW-{i:04d}" for i in range(1, len(df) + 1)])
    out["BANKRUPT"] = df["Bankrupt?"].astype(int)
    return out, name


# ------------------------------------------------------------------ الشركات البولندية
def parse_arff(text):
    """قارئ ARFF بسيط: أسماء الأعمدة من @attribute والقيم بعد @data، و«?» = قيمة مفقودة."""
    cols, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if low.startswith("@attribute"):
            cols.append(s.split()[1].strip("'\""))
        elif low.startswith("@data"):
            in_data = True
        elif in_data:
            rows.append(s.split(","))
    return pd.DataFrame(rows, columns=cols).replace("?", np.nan)


def prepare_polish(path):
    name, data = find_member(path, "5year.arff")
    df = parse_arff(data.decode("utf-8", errors="replace"))
    attrs = [f"Attr{i}" for i in range(1, 65)]
    if df.columns.tolist() != attrs + ["class"]:
        sys.exit(f"أعمدة غير متوقعة في الملف البولندي: {df.columns.tolist()[:5]}...")
    out = df[attrs].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    out.insert(0, "COMPANY_ID", [f"PL-{i:04d}" for i in range(1, len(df) + 1)])
    out["BANKRUPT"] = pd.to_numeric(df["class"].str.strip("'\" "), errors="raise").astype(int)
    if len(out) != POLISH_EXPECTED_ROWS or int(out.BANKRUPT.sum()) != POLISH_EXPECTED_BANKRUPT:
        sys.exit(f"الملف البولندي فيه {len(out)} شركة و{int(out.BANKRUPT.sum())} إفلاس، والمتوقع "
                 f"{POLISH_EXPECTED_ROWS} و{POLISH_EXPECTED_BANKRUPT}. الملف ليس الملف الأصلي.")
    return out, name


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    card_zip, comp_zip = Path(sys.argv[1]), Path(sys.argv[2])  # ملف xls/zip للأفراد، csv/zip للشركات
    for p in (card_zip, comp_zip):
        if not p.exists():
            sys.exit(f"الملف غير موجود: {p}")
    DATA_DIR.mkdir(exist_ok=True)
    cards, card_member = prepare_cards(card_zip)
    comps, comp_member = prepare_companies(comp_zip)
    cards.to_csv(CARD_CSV, index=False)
    comps.to_csv(COMPANY_CSV, index=False)
    manifest = {
        "prepared_on": date.today().isoformat(),
        "individuals": {
            "source": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "doi": "https://doi.org/10.24432/C55S3H", "license": "CC BY 4.0",
            "source_file_sha256": sha256(card_zip), "member": card_member,
            "rows": len(cards), "defaults": int(cards.DEFAULT.sum()),
        },
        "companies": {
            "source": "https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction",
            "doi": "https://doi.org/10.24432/C5004D", "license": "CC BY 4.0",
            "source_file_sha256": sha256(comp_zip), "member": comp_member,
            "rows": len(comps), "bankrupt": int(comps.BANKRUPT.sum()),
        },
    }
    if len(sys.argv) == 4:
        pl_path = Path(sys.argv[3])
        if not pl_path.exists():
            sys.exit(f"الملف غير موجود: {pl_path}")
        pol, pol_member = prepare_polish(pl_path)
        pol.to_csv(POLISH_CSV, index=False)
        manifest["companies_polish"] = {
            "source": "https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data",
            "doi": "https://doi.org/10.24432/C5F600", "license": "CC BY 4.0",
            "source_file_sha256": sha256(pl_path), "member": pol_member,
            "rows": len(pol), "bankrupt": int(pol.BANKRUPT.sum()),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
