"""
tests/test_core.py — اختبارات على البيانات الحقيقية (بدون واجهة).

التشغيل من مجلد المشروع:
    python tests/test_core.py
"""
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth  # noqa: E402
import data_loader as dl  # noqa: E402
import models  # noqa: E402
import risk  # noqa: E402

RESULTS = {}


def test(fn):
    def run():
        t = time.time()
        try:
            fn()
            RESULTS[fn.__name__] = ("PASS", time.time() - t, "")
        except Exception as e:  # noqa: BLE001
            RESULTS[fn.__name__] = ("FAIL", time.time() - t, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}")
    run.__name__ = fn.__name__
    return run


STATE = {}


@test
def test_load_real_individuals():
    df = dl.load_individuals()
    assert len(df) == 30_000, len(df)
    assert df[dl.CARD_TARGET].sum() == 6_636, df[dl.CARD_TARGET].sum()
    assert df[dl.CARD_ID].is_unique
    assert set(df.EDUCATION.unique()) <= {1, 2, 3, 4}
    STATE["ind"] = df


@test
def test_load_real_companies():
    df = dl.load_companies()
    assert len(df) == 6_819, len(df)
    assert df[dl.COMPANY_TARGET].sum() == 220, df[dl.COMPANY_TARGET].sum()
    assert "Net Income Flag" not in dl.ATTRS and len(dl.ATTRS) == 94
    assert not np.isinf(df[dl.ATTRS].to_numpy(dtype=float)).any()
    STATE["comp"] = df


@test
def test_no_leakage_and_fairness():
    for g in models.GROUPS:
        feats = models.features(g)
        assert models.GROUPS[g]["target"] not in feats
        assert models.GROUPS[g]["id"] not in feats
    assert "SEX" not in models.features("individuals") and "MARRIAGE" not in models.features("individuals")


@test
def test_train_individuals():
    res = models.train_group(STATE["ind"], "individuals")
    assert set(res["X_train"].index).isdisjoint(res["X_test"].index)
    assert len(res["X_test"]) == 9_000
    for p in res["proba_test"].values():
        assert np.all((p >= 0) & (p <= 1))
    m = models.metrics_table(res, res["base_rate"])
    assert (m["ROC-AUC"] > 0.65).all(), m
    assert (m["تحسّن Brier"] > 0).all(), m
    STATE["res_ind"] = res
    STATE["m_ind"] = m


@test
def test_train_companies():
    res = models.train_group(STATE["comp"], "companies")
    assert set(res["X_train"].index).isdisjoint(res["X_test"].index)
    m = models.metrics_table(res, res["base_rate"])
    assert (m["ROC-AUC"] > 0.65).all(), m
    STATE["res_comp"] = res
    STATE["m_comp"] = m


@test
def test_confusion_matches_test_size():
    for key in ("res_ind", "res_comp"):
        res = STATE[key]
        for kind in ("logistic", "tree"):
            cm = models.confusion(res, kind, 0.3)
            assert sum(cm.values()) == len(res["y_test"])


@test
def test_predict_and_explain_every_test_record_sample():
    for key, df, idc in (("res_ind", STATE["ind"], dl.CARD_ID), ("res_comp", STATE["comp"], dl.COMPANY_ID)):
        res = STATE[key]
        sample = df[df[idc].isin(res["ids_test"])].sample(200, random_state=0)
        for _, r in sample.iterrows():
            row = r.to_frame().T
            row[models.features(res["group"])] = row[models.features(res["group"])].apply(pd.to_numeric)
            p = models.predict(res, row)
            assert all(0 <= v <= 1 for v in p.values())
            c = models.logistic_contributions(res, row)
            assert np.isfinite(c.contribution).all()
            steps, n, rate = models.tree_path(res, row)
            assert n > 0 and 0 <= rate <= 1
        # التنبؤ داخل التطبيق يجب أن يطابق التنبؤ على مجموعة الاختبار
        first_id = res["ids_test"][0]
        row = df[df[idc] == first_id]
        assert abs(models.predict(res, row)["logistic"] - res["proba_test"]["logistic"][0]) < 1e-9


@test
def test_company_with_missing_values():
    res = STATE["res_comp"]
    row = STATE["comp"].iloc[[0]].copy()
    row[dl.ATTRS] = np.nan
    p = models.predict(res, row)
    assert all(0 <= v <= 1 for v in p.values())


@test
def test_edited_individual_raises_risk_when_late():
    res = STATE["res_ind"]
    row = STATE["ind"][STATE["ind"][dl.CARD_ID] == res["ids_test"][0]].copy()
    row[dl.PAY_COLS] = -1
    good = models.predict(res, row)["logistic"]
    row[dl.PAY_COLS] = 3
    bad = models.predict(res, row)["logistic"]
    assert bad > good, (good, bad)


@test
def test_currency_round_trip_keeps_prediction():
    """عرض المبالغ بالدينار ثم إعادتها إلى الدولار التايواني لا يغيّر التنبؤ عمليًا."""
    import currency as ui
    res = STATE["res_ind"]
    row = STATE["ind"][STATE["ind"][dl.CARD_ID] == res["ids_test"][0]].copy()
    p0 = models.predict(res, row)
    money_cols = ["LIMIT_BAL"] + dl.BILL_COLS + dl.PAYAMT_COLS
    row[money_cols] = [ui.to_ntd(round(ui.to_da(float(v)))) for v in row[money_cols].iloc[0]]
    row["UTILIZATION"] = dl.utilization(row["BILL_AMT1"], row["LIMIT_BAL"])
    p1 = models.predict(res, row)
    assert all(abs(p0[k] - p1[k]) < 0.002 for k in p0), (p0, p1)
    assert abs(ui.to_da(1000) - 4135) < 1e-9 and abs(ui.to_ntd(4135) - 1000) < 1e-9


@test
def test_risk_formulas():
    assert abs(risk.ecl(0.2, 0.45, 100_000) - 9_000) < 1e-9
    ead, drawn, undrawn = risk.card_ead(100_000, 40_000, 0.5)
    assert (ead, drawn, undrawn) == (70_000, 40_000, 60_000)
    ead, drawn, undrawn = risk.card_ead(100_000, -5_000, 0.5)   # رصيد دائن: لا يوجد مبلغ مستعمل
    assert drawn == 0 and ead == 50_000
    ead, _, undrawn = risk.card_ead(100_000, 150_000, 0.5)      # تجاوز السقف
    assert undrawn == 0 and ead == 150_000
    assert risk.risk_band(0.10, (0.15, 0.35)) == "low"
    assert risk.risk_band(0.20, (0.15, 0.35)) == "mid"
    assert risk.risk_band(0.50, (0.15, 0.35)) == "high"
    assert abs(risk.portfolio_ecl([0.1, 0.2], 0.5, [100, 200]) - 25) < 1e-9


@test
def test_decision_tables_from_case_study_doc():
    # الأفراد: أقل من 3 000 قبول، 3 000–8 000 مراجعة، أكثر من 8 000 رفض
    assert risk.decide(2_999, "individuals")[0] == "accept"
    assert risk.decide(3_000, "individuals")[0] == "review"
    assert risk.decide(8_000, "individuals")[0] == "review"
    assert risk.decide(8_001, "individuals")[0] == "reject"
    # الشركات: أقل من 15 000 قبول، 15 000–40 000 مراجعة، أكثر من 40 000 رفض
    assert risk.decide(14_999, "companies")[1] == "قبول القرض"
    assert risk.decide(40_000, "companies")[0] == "review"
    assert risk.decide(40_001, "companies")[1] == "رفض القرض"
    # حالة أحمد: حد 300 000، رصيد 0، CCF 100%، PD 5%، LGD 45% → EL = 6 750 → مراجعة
    ead, _, _ = risk.card_ead(300_000, 0, 1.0)
    assert ead == 300_000 and abs(risk.ecl(0.05, 0.45, ead) - 6_750) < 1e-9
    assert risk.safe_ratio(600_000, 400_000) == 1.5 and risk.safe_ratio(1, 0) is None and risk.safe_ratio(None, 2) is None
    try:
        risk.decide(-1, "companies")
        raise AssertionError("negative EL accepted")
    except ValueError:
        pass


@test
def test_lgd_and_ead_formulas():
    # قرض: 1 000 000 − 200 000 مسدَّد + 15 000 فوائد مستحقة = 815 000
    assert risk.loan_ead(1_000_000, 200_000, 15_000) == 815_000
    # ضمان 600 000 باقتطاع 30% = 420 000 مغطّى؛ الباقي 395 000 × 10% = 39 500؛ تكاليف 20 000 → مسترد 439 500
    r = risk.lgd_from_recovery(815_000, 600_000, 0.30, 20_000, 0.10)
    assert abs(r["secured"] - 420_000) < 1e-6 and abs(r["recovered"] - 439_500) < 1e-6
    assert abs(r["lgd"] - (1 - 439_500 / 815_000)) < 1e-12
    # بلا ضمان ولا استرداد: LGD = 100%
    assert risk.lgd_from_recovery(100_000, 0, 0, 0, 0)["lgd"] == 1.0
    # ضمان أكبر من التعرض: LGD لا تنزل تحت 0
    assert risk.lgd_from_recovery(100_000, 1_000_000, 0, 0, 0)["lgd"] == 0.0
    # تكاليف أكبر من كل المسترد: LGD لا تتجاوز 100%
    assert risk.lgd_from_recovery(100_000, 10_000, 0, 50_000, 0)["lgd"] == 1.0
    for bad in (lambda: risk.loan_ead(100, 200, 0), lambda: risk.loan_ead(-1, 0, 0),
                lambda: risk.lgd_from_recovery(0, 1, 0, 0, 0), lambda: risk.lgd_from_recovery(10, -1, 0, 0, 0),
                lambda: risk.lgd_from_recovery(10, 1, 1.5, 0, 0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("قيمة خاطئة لم تُرفض")


@test
def test_risk_input_errors():
    for bad in (lambda: risk.ecl(1.5, 0.45, 100), lambda: risk.ecl(0.1, -0.1, 100), lambda: risk.ecl(0.1, 0.4, -1),
                lambda: risk.ecl(float("nan"), 0.4, 1), lambda: risk.card_ead(0, 10, 0.5),
                lambda: risk.card_ead(100, 10, 2), lambda: risk.risk_band(0.2, (0.4, 0.3))):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("قيمة خاطئة لم تُرفض")


@test
def test_auth():
    rec = auth.make_record("owner", "Strong-Pass-123")
    assert "Strong-Pass-123" not in str(rec)
    assert auth.verify("owner", "Strong-Pass-123", rec)
    assert auth.verify(" owner ", "Strong-Pass-123", rec)
    assert not auth.verify("owner", "wrong", rec)
    assert not auth.verify("other", "Strong-Pass-123", rec)
    assert not auth.verify("owner", "x", None)
    assert not auth.verify("owner", "x", {"username": "owner"})
    assert not auth.verify("owner", "x", {**rec, "salt": "not-hex"})
    assert auth.to_toml(rec).startswith("[auth]\n")
    import tomllib
    assert tomllib.loads(auth.to_toml(rec))["auth"] == rec


@test
def test_data_errors_are_clear():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            dl.load_individuals(Path(tmp) / "missing.csv")
            raise AssertionError("no error")
        except dl.DataError as e:
            assert "غير موجود" in str(e)
        bad = Path(tmp) / "bad.csv"
        pd.DataFrame({"ID": [1], "LIMIT_BAL": [1]}).to_csv(bad, index=False)
        try:
            dl.load_individuals(bad)
            raise AssertionError("no error")
        except dl.DataError as e:
            assert "ناقص" in str(e)
        try:
            dl.load_companies(bad)
            raise AssertionError("no error")
        except dl.DataError as e:
            assert "ناقص" in str(e)


if __name__ == "__main__":
    for t in [test_load_real_individuals, test_load_real_companies, test_no_leakage_and_fairness, test_train_individuals,
              test_train_companies, test_confusion_matches_test_size, test_predict_and_explain_every_test_record_sample,
              test_company_with_missing_values, test_edited_individual_raises_risk_when_late,
              test_currency_round_trip_keeps_prediction, test_risk_formulas, test_decision_tables_from_case_study_doc, test_lgd_and_ead_formulas,
              test_risk_input_errors, test_auth, test_data_errors_are_clear]:
        t()
        status, sec, err = RESULTS[t.__name__]
        print(f"{status}  {t.__name__:<55} {sec:6.1f}s")
        if err:
            print("      " + err.replace("\n", "\n      "))
    for key in ("m_ind", "m_comp"):
        if key in STATE:
            print(f"\n{key}:\n" + STATE[key].drop(columns="kind").round(4).to_string(index=False))
    fails = sum(1 for s, _, _ in RESULTS.values() if s == "FAIL")
    print(f"\n{len(RESULTS) - fails} PASS, {fails} FAIL")
    sys.exit(1 if fails else 0)
