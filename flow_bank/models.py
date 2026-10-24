"""
models.py — تدريب نموذجين (الانحدار اللوجستي وشجرة القرار) وتقييمهما وتفسيرهما.

القواعد:
- نقسم البيانات 70% تدريب / 30% اختبار مع الحفاظ على نسبة المتعثرين (stratify).
- نختار إعدادات كل نموذج بالتحقق المتقاطع (5 أجزاء) على بيانات التدريب فقط.
- كل المقاييس المعروضة محسوبة على مجموعة الاختبار التي لم يرها النموذج.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

import data_loader as dl

RANDOM_STATE = 42
TEST_SIZE = 0.30
MODEL_NAMES = {"logistic": "الانحدار اللوجستي", "tree": "شجرة القرار"}


# ======================================================================= معالجة المتغيرات
def signed_log(x):
    """لوغاريتم يحافظ على الإشارة: يقلّل أثر المبالغ الضخمة دون حذفها (مفيد للانحدار اللوجستي)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class Winsorizer(BaseEstimator, TransformerMixin):
    """يقصّ القيم المتطرفة إلى المئين 1% و99% المحسوبين من بيانات التدريب فقط."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


GROUPS = {
    "individuals": {
        "name": "الأفراد",
        "id": dl.CARD_ID,
        "target": dl.CARD_TARGET,
        "money": ["LIMIT_BAL"] + dl.BILL_COLS + dl.PAYAMT_COLS,
        "plain": ["AGE", "UTILIZATION"] + dl.PAY_COLS,
        "cat": ["EDUCATION"],
        "grid_lr": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "grid_tree": {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [50, 200]},
    },
    "companies": {
        "name": "الشركات",
        "id": dl.COMPANY_ID,
        "target": dl.COMPANY_TARGET,
        "money": [],
        "plain": dl.ATTRS,
        "cat": [],
        "grid_lr": {"model__C": [0.01, 0.1, 1.0]},
        "grid_tree": {"model__max_depth": [3, 4, 5, 6, 8], "model__min_samples_leaf": [10, 30]},
    },
    "companies_pl": {
        "name": "الشركات (نسب مالية — بولندا)",
        "id": dl.COMPANY_ID,
        "target": dl.COMPANY_TARGET,
        "money": [],
        "plain": dl.POLISH_ATTRS,
        "cat": [],
        "grid_lr": {"model__C": [0.01, 0.1, 1.0]},
        "grid_tree": {"model__max_depth": [3, 4, 5, 6], "model__min_samples_leaf": [10, 30]},
    },
}


def features(group):
    g = GROUPS[group]
    return g["money"] + g["plain"] + g["cat"]


def _prep(group, for_tree):
    g = GROUPS[group]
    parts = []
    if group == "individuals":
        if for_tree:
            # الشجرة لا تحتاج تحويلًا: تبقى شروطها بالقيم الأصلية (دولار تايواني، أشهر تأخر...).
            parts.append(("num", "passthrough", g["money"] + g["plain"]))
        else:
            parts.append(("money", Pipeline([("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")),
                                             ("scale", StandardScaler())]), g["money"]))
            parts.append(("plain", StandardScaler(), g["plain"]))
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), g["cat"]))
    else:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if not for_tree:
            steps = [("winsor", Winsorizer()), ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        parts.append(("ratios", Pipeline(steps), g["plain"]))
    return ColumnTransformer(parts)


def _pipeline(group, kind):
    if kind == "logistic":
        model = LogisticRegression(max_iter=5000)
    else:
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    return Pipeline([("prep", _prep(group, kind == "tree")), ("model", model)])


# ======================================================================= التدريب
def train_group(df, group):
    g = GROUPS[group]
    X, y = df[features(group)], df[g["target"]].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipes, best, cv_auc, proba, seconds = {}, {}, {}, {}, {}
    for kind, grid in (("logistic", g["grid_lr"]), ("tree", g["grid_tree"])):
        t = time.time()
        search = GridSearchCV(_pipeline(group, kind), grid, scoring="roc_auc", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        pipes[kind] = search.best_estimator_
        best[kind] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        cv_auc[kind] = float(search.best_score_)
        proba[kind] = pipes[kind].predict_proba(X_test)[:, 1]
        seconds[kind] = time.time() - t
    return {
        "group": group, "pipes": pipes, "best_params": best, "cv_auc": cv_auc, "seconds": seconds,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "ids_test": df.loc[X_test.index, g["id"]].to_numpy(),
        "proba_test": proba, "base_rate": float(y_train.mean()),
    }


# ======================================================================= التقييم
def metrics_table(res, threshold):
    y = res["y_test"].to_numpy()
    base = res["base_rate"]
    brier_ref = brier_score_loss(y, np.full(len(y), base))
    rows = []
    for kind, p in res["proba_test"].items():
        pred = (p >= threshold).astype(int)
        brier = brier_score_loss(y, p)
        rows.append({
            "kind": kind, "النموذج": MODEL_NAMES[kind],
            "ROC-AUC": roc_auc_score(y, p),
            "Recall": recall_score(y, pred, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "Brier Score": brier,
            "Brier (نموذج ساذج)": brier_ref,
            "تحسّن Brier": 1 - brier / brier_ref,
            "ROC-AUC (تحقق متقاطع)": res["cv_auc"][kind],
        })
    return pd.DataFrame(rows)


def confusion(res, kind, threshold):
    pred = (res["proba_test"][kind] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(res["y_test"], pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def roc_points(res):
    frames = []
    for kind, p in res["proba_test"].items():
        fpr, tpr, _ = roc_curve(res["y_test"], p)
        frames.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": MODEL_NAMES[kind]}))
    return pd.concat(frames, ignore_index=True)


def calibration_points(res, bins=10):
    frames = []
    for kind, p in res["proba_test"].items():
        frac, mean = calibration_curve(res["y_test"], p, n_bins=bins, strategy="quantile")
        frames.append(pd.DataFrame({"predicted": mean, "observed": frac, "model": MODEL_NAMES[kind]}))
    return pd.concat(frames, ignore_index=True)


# ======================================================================= التنبؤ والتفسير
def predict(res, row):
    """row: DataFrame بسطر واحد يحتوي متغيرات التنبؤ. يعيد احتمال التعثر من كل نموذج."""
    X = row[features(res["group"])]
    return {kind: float(pipe.predict_proba(X)[0, 1]) for kind, pipe in res["pipes"].items()}


def _original_name(transformed):
    """يحوّل اسم العمود بعد المعالجة (مثل cat__EDUCATION_2) إلى اسم المتغير الأصلي."""
    name = transformed.split("__", 1)[-1]
    if name.startswith("EDUCATION_"):
        return "EDUCATION"
    return name


def logistic_contributions(res, row):
    """مساهمة كل متغير = المعامل × القيمة الموحَّدة: موجبة ترفع الاحتمال مقارنة بمقترض متوسط."""
    pipe = res["pipes"]["logistic"]
    prep = pipe.named_steps["prep"]
    xt = prep.transform(row[features(res["group"])])[0]
    coef = pipe.named_steps["model"].coef_[0]
    names = [_original_name(n) for n in prep.get_feature_names_out()]
    out = pd.DataFrame({"feature": names, "contribution": xt * coef}).groupby("feature", as_index=False).sum()
    out["abs"] = out.contribution.abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def tree_path(res, row):
    """المسار الفعلي للسجل داخل الشجرة المدرَّبة."""
    pipe = res["pipes"]["tree"]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["model"]
    xt = prep.transform(row[features(res["group"])])
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    t = clf.tree_
    leaf = int(clf.apply(xt)[0])
    steps = []
    for node in clf.decision_path(xt).indices:
        if node == leaf:
            continue
        f = int(t.feature[node])
        steps.append({"feature": names[f], "threshold": float(t.threshold[node]), "value": float(xt[0, f]),
                      "left": bool(xt[0, f] <= t.threshold[node]), "samples": int(t.n_node_samples[node])})
    return steps, int(t.n_node_samples[leaf]), float(clf.predict_proba(xt)[0, 1])


def tree_importance(res):
    pipe = res["pipes"]["tree"]
    names = [_original_name(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.DataFrame({"feature": names, "importance": pipe.named_steps["model"].feature_importances_})
    return imp.groupby("feature", as_index=False).sum().sort_values("importance", ascending=False)


def key_features(res, n=8):
    """أهم المتغيرات: من أهمية الشجرة أولًا، ثم من أكبر معاملات الانحدار اللوجستي."""
    ranked = [f for f, v in tree_importance(res).itertuples(index=False) if v > 0]
    pipe = res["pipes"]["logistic"]
    names = [_original_name(x) for x in pipe.named_steps["prep"].get_feature_names_out()]
    coef = pd.Series(np.abs(pipe.named_steps["model"].coef_[0]), index=names).groupby(level=0).sum()
    for f in coef.sort_values(ascending=False).index:
        if f not in ranked:
            ranked.append(f)
    return ranked[:n]
