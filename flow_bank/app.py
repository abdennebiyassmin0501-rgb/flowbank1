"""
app.py — Flow Bank: منصة عربية لتقييم مخاطر الائتمان ببيانات حقيقية من UCI.

التشغيل:  streamlit run app.py
"""
import time

import numpy as np
import pandas as pd
import streamlit as st

import auth
import data_loader as dl
import models
import risk
import ui

st.set_page_config(page_title="Flow Bank", page_icon="🏦", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()

PAGES = ["الرئيسية", "تقييم فرد", "تقييم شركة", "تقييم LGD وEAD", "أداء النماذج"]


# ======================================================================= البيانات والنماذج
@st.cache_data(show_spinner=False)
def get_data():
    return dl.load_individuals(), dl.load_companies()


@st.cache_resource(show_spinner=False)
def get_models():
    ind, comp = get_data()
    return {"individuals": models.train_group(ind, "individuals"),
            "companies": models.train_group(comp, "companies")}


# ======================================================================= تسجيل الدخول
def load_auth_config():
    try:
        if "auth" in st.secrets:
            return dict(st.secrets["auth"])
    except Exception:  # لا يوجد ملف أسرار إطلاقًا
        pass
    return None


def login_page():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        _login_box()


def _login_box():
    st.write("")
    st.markdown('<div class="fb-bar"><div class="fb-brand"><div class="fb-logo"></div><div>'
                '<div class="fb-name">Flow Bank</div><div class="fb-sub">تقييم مخاطر الائتمان</div></div></div></div>',
                unsafe_allow_html=True)
    cfg = load_auth_config()
    if not auth.valid_config(cfg):
        st.error("حساب الدخول غير مضبوط. محليًا: python set_password.py · على Streamlit Cloud: قسم [auth] في Settings ← Secrets.")
        return
    wait = st.session_state.get("lock_until", 0) - time.time()
    if wait > 0:
        st.error(f"الدخول موقوف مؤقتًا بعد محاولات خاطئة. المدة المتبقية: {int(wait) + 1} ثانية.")
        return
    with st.form("login"):
        st.markdown("#### تسجيل الدخول")
        username = st.text_input("اسم المستخدم", key="lg_user")
        password = st.text_input("كلمة المرور", type="password", key="lg_pass")
        ok = st.form_submit_button("دخول", type="primary", use_container_width=True)
    if ok:
        if auth.verify(username, password, cfg):
            st.session_state.update(auth=True, user=username.strip(), fails=0)
            st.rerun()
        fails = st.session_state.get("fails", 0) + 1
        st.session_state["fails"] = fails
        if fails >= 5:
            st.session_state.update(lock_until=time.time() + 60, fails=0)
            st.error("5 محاولات خاطئة. أُوقف الدخول لمدة دقيقة.")
        else:
            st.error(f"اسم المستخدم أو كلمة المرور غير صحيحة ({fails} من 5).")


# ======================================================================= الصفحات
def header():
    st.markdown(
        '<div class="fb-bar"><div class="fb-brand"><div class="fb-logo"></div><div>'
        '<div class="fb-name">Flow Bank</div><div class="fb-sub">تقييم مخاطر الائتمان ببيانات حقيقية</div></div></div>'
        f'<div class="fb-user">{st.session_state.get("user", "")}</div></div>', unsafe_allow_html=True)
    nav, out = st.columns([6, 1])
    with nav:
        page = st.radio("التنقل", PAGES, horizontal=True, key="page", label_visibility="collapsed")
    with out:
        if st.button("خروج", key="logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return page


def page_home(ind, comp, res):
    st.markdown(
        '<div class="fb-hero"><h1>تقييم مخاطر الائتمان</h1>'
        '<p>تقيّم Flow Bank طلبات الأفراد والشركات بالخسارة المتوقعة EL = PD × LGD × EAD وجدول قرار واضح، '
        'وتعرض أداء نموذجين مدرَّبين على بيانات حقيقية من UCI Machine Learning Repository.</p></div>',
        unsafe_allow_html=True)
    st.write("")
    mi = models.metrics_table(res["individuals"], res["individuals"]["base_rate"])
    mc = models.metrics_table(res["companies"], res["companies"]["base_rate"])
    c = st.columns(4)
    c[0].markdown(ui.kpi("عملاء بطاقات الائتمان", f"{len(ind):,}".replace(",", " "), "تايوان، أبريل–سبتمبر 2005"), unsafe_allow_html=True)
    c[1].markdown(ui.kpi("نسبة التعثر الفعلية", ui.pct(ind[dl.CARD_TARGET].mean()), "تعثّر في الشهر التالي"), unsafe_allow_html=True)
    c[2].markdown(ui.kpi("شركات تايوانية", f"{len(comp):,}".replace(",", " "), "1999–2009"), unsafe_allow_html=True)
    c[3].markdown(ui.kpi("نسبة الإفلاس الفعلية", ui.pct(comp[dl.COMPANY_TARGET].mean()), "حسب قواعد بورصة تايوان"), unsafe_allow_html=True)

    st.subheader("دلالة الألوان")
    st.markdown(
        f"{ui.tag('real')} قيمة من البيانات الأصلية. &nbsp; "
        f"{ui.tag('model')} احتمال صادر عن نموذج مدرَّب. &nbsp; "
        f"{ui.tag('assume')} افتراض توضيحي قابل للتعديل. &nbsp; "
        f"{ui.tag('input')} قيمة مُدخلة يدويًا.", unsafe_allow_html=True)

    st.subheader("الخدمات")
    a, b, d, c3 = st.columns(4)
    with a:
        st.markdown('<div class="fb-card"><h4>تقييم فرد</h4><p class="small">تقدير الخسارة المتوقعة والقرار الائتماني لطلب بطاقة، '
                    'انطلاقًا من الدخل والمصاريف والحد المطلوب والرصيد.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="fb-card"><h4>تقييم شركة</h4><p class="small">تحليل القوائم المالية ومبلغ القرض: '
                    'النسب المالية، الخسارة المتوقعة، والقرار الائتماني.</p></div>', unsafe_allow_html=True)
    with d:
        st.markdown('<div class="fb-card"><h4>تقييم LGD وEAD</h4><p class="small">حساب التعرض عند التعثر للبطاقة أو القرض، '
                    'والخسارة عند التعثر من الضمان وتكاليف التحصيل.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="fb-card"><h4>أداء النماذج</h4><p class="small">مقارنة النموذجين على بيانات اختبار '
                    'مستقلة، بمقاييس التمييز والدقة ومصفوفة الالتباس.</p></div>', unsafe_allow_html=True)

    st.subheader("أفضل أداء على بيانات الاختبار")
    x, y = st.columns(2)
    for col, m, name in ((x, mi, "الأفراد"), (y, mc, "الشركات")):
        best = m.loc[m["ROC-AUC"].idxmax()]
        col.markdown(ui.kpi(f"{name}: {best['النموذج']}", f"{best['ROC-AUC']:.3f}",
                            "مقياس التمييز على بيانات الاختبار (الأعلى أفضل، والأقصى 1)"), unsafe_allow_html=True)

    st.subheader("مصادر البيانات وحدودها")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(
            '<div class="fb-card"><h4>الأفراد: Default of Credit Card Clients</h4>'
            '<p class="small">I-Cheng Yeh (2009)، UCI #350، رخصة CC BY 4.0. '
            '<a href="https://doi.org/10.24432/C55S3H" target="_blank">doi.org/10.24432/C55S3H</a></p>'
            '<ul class="small"><li>30 000 حامل بطاقة في تايوان. المبالغ الأصلية بالدولار التايواني، معروضة بالدينار '
            '(\u20661 NT$ = 4.135 DA\u2069، سعر أكتوبر 2026).</li>'
            '<li>التعثر هنا = عدم السداد في <b>الشهر التالي</b>، وليس خلال 12 شهرًا.</li>'
            '<li>الجنس والحالة الاجتماعية مستبعدان من التنبؤ تجنّبًا للتمييز.</li></ul></div>',
            unsafe_allow_html=True)
    with s2:
        st.markdown(
            '<div class="fb-card"><h4>الشركات: Taiwanese Bankruptcy Prediction</h4>'
            '<p class="small">UCI #572 (2020)، مصدرها Taiwan Economic Journal، رخصة CC BY 4.0. '
            '<a href="https://doi.org/10.24432/C5004D" target="_blank">doi.org/10.24432/C5004D</a></p>'
            '<ul class="small"><li>6 819 شركة تايوانية (1999–2009)، 95 مؤشرًا ماليًا، منها 220 شركة أفلست.</li>'
            '<li>الحدث المتوقَّع هو <b>الإفلاس</b> حسب قواعد بورصة تايوان، والمصدر لا يحدد أفقًا زمنيًا للتنبؤ.</li>'
            '<li>أغلب المؤشرات مُقيَّسة من المصدر بين 0 و1، وبلا مبالغ قروض، لذلك يُدخَل مبلغ التعرض للشركات يدويًا.</li></ul></div>',
            unsafe_allow_html=True)


# ---------------------------------------------------------------------- نماذج دراسة الحالة
# الحقول مأخوذة من وثيقة «دراسات حالة: اتخاذ القرار الائتماني». النموذج فارغ: المستخدمة تُدخل كل القيم.
MARITAL = ["أعزب/عزباء", "متزوج/ة", "مطلق/ة", "أرمل/ة"]
HISTORY = ["لا يوجد", "جيد (بدون تأخر)", "تأخرات سابقة"]
GUARANTEE_IND = ["لا يوجد", "كفيل", "رهن أو ضمان عيني", "أخرى"]
GUARANTEE_COMP = ["لا يوجد", "معدات ومخزون", "عقار", "كفالة بنكية", "أخرى"]
DECISION_STYLE = {"accept": "low", "review": "mid", "reject": "high"}


def money_input(col, label, key, min_value=0.0):
    return col.number_input(f"{label} (DA)", min_value=min_value, value=None, step=1_000.0, format="%.0f",
                            key=key, placeholder="0")


def missing_fields(values):
    return [name for name, v in values.items() if v is None or v == ""]


def decision_block(el, group, pd_, lgd, ead, ead_note):
    key, text = risk.decide(el, group)
    k = st.columns(4)
    k[0].markdown(ui.kpi(f"PD {ui.tag('input')}", ui.pct(pd_), "احتمال التعثر المُدخل"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi(f"LGD {ui.tag('assume')}", ui.pct(lgd, 0), "الخسارة عند التعثر"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("EAD", ui.money(ead, convert=False), ead_note), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("EL", ui.money(el, convert=False), "الخسارة المتوقعة"), unsafe_allow_html=True)
    st.markdown(f'<div class="ecl-eq">EL = {ui.pct(pd_)} × {ui.pct(lgd, 0)} × {ui.money(ead, convert=False)} = '
                f'<b>{ui.money(el, convert=False)}</b></div>', unsafe_allow_html=True)
    st.markdown(f"#### القرار المقترح: {ui.pill(DECISION_STYLE[key], {DECISION_STYLE[key]: text})}", unsafe_allow_html=True)
    low, high = risk.DECISION_TABLES[group]["limits"]
    texts = risk.DECISION_TABLES[group]["texts"]
    da = lambda v: ui.money(v, convert=False)  # مبلغ معزول الاتجاه حتى لا ينقلب داخل النص العربي
    table = pd.DataFrame({"EL": [f"أقل من {da(low)}", f"من {da(low)} إلى {da(high)}", f"أكثر من {da(high)}"],
                          "القرار": [texts["accept"], texts["review"], texts["reject"]]})
    with st.expander("جدول اتخاذ القرار (من وثيقة دراسات الحالة)"):
        st.dataframe(table, hide_index=True, use_container_width=True)


def ratio_kpi(label, value, sub, as_pct=True):
    shown = "غير محسوبة" if value is None else (ui.pct(value) if as_pct else ui.fmt_num(value, 2))
    return ui.kpi(label, shown, sub)


def page_individual(ind):
    st.title("تقييم فرد")
    st.caption("جميع الحقول إلزامية. المبالغ بالدينار الجزائري (DA).")
    st.markdown("### بيانات العميل")
    a, b, c = st.columns(3)
    name = a.text_input("الاسم", key="i_name", placeholder="اسم العميل")
    age = b.number_input("العمر (سنة)", min_value=18, max_value=100, value=None, step=1, key="i_age", placeholder="مثال: 32")
    marital = c.selectbox("الحالة الاجتماعية", MARITAL, index=None, key="i_marital", placeholder="—")
    a, b, c = st.columns(3)
    job = a.text_input("المهنة", key="i_job", placeholder="مثال: موظف في شركة خاصة")
    years = b.number_input("مدة العمل (سنة)", min_value=0.0, max_value=60.0, value=None, step=0.5, key="i_years", placeholder="مثال: 3")
    history = c.selectbox("سجل ائتماني", HISTORY, index=None, key="i_hist", placeholder="—")
    a, b, c = st.columns(3)
    income = money_input(a, "الدخل الشهري", "i_income")
    expenses = money_input(b, "المصاريف الشهرية", "i_exp")
    limit = money_input(c, "الحد الائتماني المطلوب", "i_limit")
    a, b, c = st.columns(3)
    balance = money_input(a, "الرصيد الحالي", "i_bal")
    guarantee = b.selectbox("الضمان", GUARANTEE_IND, index=None, key="i_guar", placeholder="—")

    st.markdown("### معطيات الحساب")
    a, b, c = st.columns(3)
    pd_pct = a.number_input("PD: احتمال التعثر (%)", min_value=0.0, max_value=100.0, value=None, step=0.5, key="i_pd",
                            placeholder="مثال: 5")
    lgd = b.slider("LGD: نسبة الخسارة عند التعثر (%)", 0, 100, int(risk.DEFAULT_LGD * 100), 5, key="i_lgd", format="%d%%") / 100
    ccf = c.slider("CCF: نسبة استعمال الجزء غير المستعمل من الحد (%)", 0, 100, 100, 5, key="i_ccf", format="%d%%") / 100
    st.caption(f"مرجع {ui.tag('real')}: معدل التعثر في بيانات UCI {ui.pct(ind[dl.CARD_TARGET].mean())} (الشهر التالي).",
               unsafe_allow_html=True)

    required = {"الاسم": name.strip(), "العمر": age, "الحالة الاجتماعية": marital, "المهنة": job.strip(),
                "مدة العمل": years, "الدخل الشهري": income, "المصاريف الشهرية": expenses,
                "الحد الائتماني المطلوب": limit, "الرصيد الحالي": balance, "الضمان": guarantee,
                "سجل ائتماني": history, "PD": pd_pct}
    missing = missing_fields(required)
    if missing:
        st.info("حقول ناقصة: " + "، ".join(missing) + ".")
        return
    if limit <= 0:
        st.error("الحد الائتماني المطلوب يجب أن يكون أكبر من صفر.")
        return
    if income <= 0:
        st.error("الدخل الشهري يجب أن يكون أكبر من صفر.")
        return

    st.markdown(f"### النتيجة: {name.strip()}")
    pd_ = pd_pct / 100
    ead, drawn, undrawn = risk.card_ead(limit, balance, ccf)
    el = risk.ecl(pd_, lgd, ead)
    decision_block(el, "individuals", pd_, lgd, ead,
                   f"{ui.money(drawn, convert=False)} مستعمل + {ui.pct(ccf, 0)} × {ui.money(undrawn, convert=False)}")

    st.markdown("#### مؤشرات مساعدة من بياناتك")
    st.caption("مؤشرات تحليلية لا تدخل في حساب EL.")
    k = st.columns(4)
    k[0].markdown(ratio_kpi("المصاريف ÷ الدخل", risk.safe_ratio(expenses, income), "كم يذهب من الدخل للمصاريف"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("الدخل المتبقي شهريًا", ui.money(income - expenses, convert=False), "الدخل − المصاريف"), unsafe_allow_html=True)
    k[2].markdown(ratio_kpi("الحد ÷ الدخل السنوي", risk.safe_ratio(limit, income * 12), "حجم الحد مقارنة بالدخل"), unsafe_allow_html=True)
    k[3].markdown(ratio_kpi("الرصيد ÷ الحد", risk.safe_ratio(balance, limit), "نسبة الاستعمال الحالية"), unsafe_allow_html=True)
    notes = []
    if expenses > income:
        notes.append("المصاريف أكبر من الدخل.")
    if history == "تأخرات سابقة":
        notes.append("للعميل تأخرات سابقة في السداد.")
    if history == "لا يوجد":
        notes.append("لا يوجد سجل ائتماني: البنك لا يملك تاريخ سداد يحكم به.")
    if guarantee == "لا يوجد":
        notes.append("لا يوجد ضمان: الخسارة عند التعثر قد تكون أعلى.")
    if notes:
        st.warning(" ".join(notes))
    st.caption("الحالة الاجتماعية والمهنة للتوثيق فقط ولا تدخلان في الحساب.")


def page_company(comp):
    st.title("تقييم شركة")
    st.caption("جميع الحقول إلزامية. المبالغ بالدينار الجزائري (DA).")
    st.markdown("### بيانات الشركة")
    a, b, c = st.columns(3)
    name = a.text_input("اسم الشركة", key="c_name", placeholder="مثال: شركة تصنيع صغيرة")
    loan = money_input(b, "مبلغ القرض المطلوب", "c_loan")
    term = c.number_input("المدة (شهر)", min_value=1, max_value=360, value=None, step=1, key="c_term", placeholder="مثال: 36")
    a, b, c = st.columns(3)
    assets = money_input(a, "إجمالي الأصول", "c_assets")
    debt = money_input(b, "إجمالي الدين", "c_debt")
    cur_assets = money_input(c, "الأصول المتداولة", "c_ca")
    a, b, c = st.columns(3)
    cur_liab = money_input(a, "الالتزامات المتداولة", "c_cl")
    profit = b.number_input("صافي الربح (DA)", value=None, step=1_000.0, format="%.0f", key="c_profit",
                            placeholder="سالب إذا كانت خسارة")
    ebit = c.number_input("EBIT: الربح قبل الفوائد والضرائب (DA)", value=None, step=1_000.0, format="%.0f", key="c_ebit",
                          placeholder="0")
    a, b, _ = st.columns(3)
    interest = money_input(a, "الفوائد", "c_int")
    guarantee = b.selectbox("الضمان", GUARANTEE_COMP, index=None, key="c_guar", placeholder="—")

    st.markdown("### معطيات الحساب")
    a, b = st.columns(2)
    pd_pct = a.number_input("PD: احتمال التعثر (%)", min_value=0.0, max_value=100.0, value=None, step=0.5, key="c_pd",
                            placeholder="مثال: 3")
    lgd = b.slider("LGD: نسبة الخسارة عند التعثر (%)", 0, 100, int(risk.DEFAULT_LGD * 100), 5, key="c_lgd", format="%d%%") / 100
    st.caption(f"مرجع {ui.tag('real')}: نسبة الإفلاس في بيانات UCI {ui.pct(comp[dl.COMPANY_TARGET].mean())}. "
               "التعرض عند التعثر (EAD) يساوي مبلغ القرض.", unsafe_allow_html=True)

    required = {"اسم الشركة": name.strip(), "مبلغ القرض المطلوب": loan, "المدة": term, "إجمالي الأصول": assets,
                "إجمالي الدين": debt, "الأصول المتداولة": cur_assets, "الالتزامات المتداولة": cur_liab,
                "صافي الربح": profit, "EBIT": ebit, "الفوائد": interest, "الضمان": guarantee, "PD": pd_pct}
    missing = missing_fields(required)
    if missing:
        st.info("حقول ناقصة: " + "، ".join(missing) + ".")
        return
    if loan <= 0:
        st.error("مبلغ القرض المطلوب يجب أن يكون أكبر من صفر.")
        return
    if cur_assets > assets:
        st.error("الأصول المتداولة أكبر من إجمالي الأصول.")
        return

    st.markdown(f"### النتيجة: {name.strip()}")
    pd_ = pd_pct / 100
    el = risk.ecl(pd_, lgd, loan)
    decision_block(el, "companies", pd_, lgd, loan, f"مبلغ القرض على {int(term)} شهرًا")

    st.markdown("#### النسب المالية من بياناتك")
    st.caption("نسب تحليلية لا تدخل في حساب EL.")
    k = st.columns(3)
    k[0].markdown(ratio_kpi("نسبة المديونية", risk.safe_ratio(debt, assets), "إجمالي الدين ÷ إجمالي الأصول"), unsafe_allow_html=True)
    k[1].markdown(ratio_kpi("نسبة السيولة الجارية", risk.safe_ratio(cur_assets, cur_liab),
                            "الأصول المتداولة ÷ الالتزامات المتداولة", as_pct=False), unsafe_allow_html=True)
    k[2].markdown(ratio_kpi("تغطية الفوائد", risk.safe_ratio(ebit, interest), "EBIT ÷ الفوائد (عدد المرات)", as_pct=False),
                  unsafe_allow_html=True)
    k = st.columns(3)
    k[0].markdown(ratio_kpi("العائد على الأصول", risk.safe_ratio(profit, assets), "صافي الربح ÷ إجمالي الأصول"), unsafe_allow_html=True)
    k[1].markdown(ratio_kpi("القرض ÷ الأصول", risk.safe_ratio(loan, assets), "حجم القرض مقارنة بالأصول"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("القسط الشهري التقريبي", ui.money(loan / term, convert=False), "المبلغ ÷ المدة، بدون فوائد"),
                  unsafe_allow_html=True)
    notes = []
    cr = risk.safe_ratio(cur_assets, cur_liab)
    if cr is not None and cr < 1:
        notes.append("الأصول المتداولة أقل من الالتزامات المتداولة.")
    icr = risk.safe_ratio(ebit, interest)
    if icr is not None and icr < 1:
        notes.append("الربح قبل الفوائد لا يغطي الفوائد.")
    if profit < 0:
        notes.append("الشركة تحقق خسارة.")
    if guarantee == "لا يوجد":
        notes.append("لا يوجد ضمان: الخسارة عند التعثر قد تكون أعلى.")
    if notes:
        st.warning(" ".join(notes))


# ---------------------------------------------------------------------- أداء النماذج
def page_performance(res):
    st.title("أداء النماذج")
    group = st.radio("الفئة", ["individuals", "companies"], horizontal=True, key="perf_group",
                     format_func=lambda g: models.GROUPS[g]["name"])
    R = res[group]
    pos = "متعثر" if group == "individuals" else "مفلس"
    n_test, n_pos = len(R["y_test"]), int(R["y_test"].sum())
    st.caption(f"تدريب: {len(R['y_train']):,} سجل · اختبار: {n_test:,} سجل منها {n_pos:,} {pos}. "
               "الإعدادات اختيرت بالتحقق المتقاطع على بيانات التدريب فقط.")
    thr = st.slider("عتبة التصنيف (احتمال يُصنَّف فوقه السجل «" + pos + "»)",
                    0.01, 0.99, round(R["base_rate"], 2), 0.01, key=f"thr_{group}")
    st.caption(f"القيمة الافتراضية = نسبة الحالات «{pos}» في بيانات التدريب ({ui.pct(R['base_rate'])}). "
               "رفع العتبة يرفع Precision ويخفض Recall، والعكس. ROC-AUC وBrier لا يتأثران بالعتبة.")

    m = models.metrics_table(R, thr)
    show = m.drop(columns="kind").copy()
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "النموذج"}), hide_index=True,
                 use_container_width=True)
    best_auc = m.loc[m["ROC-AUC"].idxmax(), "النموذج"]
    best_brier = m.loc[m["Brier Score"].idxmin(), "النموذج"]
    st.markdown(f"- أفضل تمييز بين الحالات (ROC-AUC): **{best_auc}**.\n"
                f"- أدق احتمالات (أقل Brier): **{best_brier}**.\n"
                f"- «Brier (نموذج ساذج)» = خطأ من يعطي الجميع الاحتمال نفسه ({ui.pct(R['base_rate'])}). "
                "«تحسّن Brier» الموجب يعني أن النموذج أفضل من هذا التخمين.")

    st.subheader("مصفوفات الالتباس")
    c = st.columns(2)
    for col, kind in zip(c, ("logistic", "tree")):
        with col:
            st.markdown(f"**{models.MODEL_NAMES[kind]}** (العتبة {ui.pct(thr, 0)})")
            st.markdown(ui.confusion_html(models.confusion(R, kind, thr), pos=pos), unsafe_allow_html=True)
    st.caption(f"الخلايا الحمراء أخطاء: «فعلًا {pos} / توقّع سليم» خسارة مالية فاتت البنك، "
               "و«فعلًا سليم / توقّع " + pos + "» عميل جيد قد يُرفض.")

    a, b = st.columns(2)
    with a:
        st.subheader("منحنى ROC")
        ui.line_chart(models.roc_points(R), "fpr", "tpr", "model", "نسبة الإنذارات الخاطئة", "نسبة الحالات المكتشفة")
        st.caption("كلما اقترب المنحنى من الزاوية العليا كان التمييز أفضل؛ المتقطع = تخمين عشوائي.")
    with b:
        st.subheader("منحنى المعايرة")
        ui.line_chart(models.calibration_points(R), "predicted", "observed", "model", "الاحتمال المتوقَّع",
                      "النسبة الفعلية", points=True)
        st.caption("على الخط المتقطع = الاحتمال يطابق الواقع (مثلًا 20% متوقَّع ← 20% حدث فعلًا). هذا ما يقيسه Brier.")

    st.subheader("معنى المقاييس")
    st.markdown(
        f"- **ROC-AUC:** قدرة النموذج على وضع الحالات «{pos}» قبل السليمة في الترتيب. 0.5 = عشوائي، 1 = كامل.\n"
        f"- **Recall:** من بين كل {pos} فعلًا، كم اكتشف النموذج؟\n"
        f"- **Precision:** من بين من توقّع النموذج أنه {pos}، كم كان كذلك فعلًا؟\n"
        "- **Brier Score:** متوسط مربع الفرق بين الاحتمال والنتيجة (0 أو 1). كلما صغر كانت الاحتمالات أدق؛ "
        "مهم لأن ECL تستعمل الاحتمال نفسه.\n"
        "- **مصفوفة الالتباس:** عدد الإصابات والأخطاء بنوعيها عند العتبة المختارة.")
    with st.expander("إعدادات النموذجين المختارة"):
        for kind in ("logistic", "tree"):
            st.markdown(f"**{models.MODEL_NAMES[kind]}:** {R['best_params'][kind]} · "
                        f"ROC-AUC في التحقق المتقاطع {R['cv_auc'][kind]:.3f} · مدة التدريب {R['seconds'][kind]:.1f} ثانية")


# ---------------------------------------------------------------------- تقييم LGD وEAD
FACILITY = ["بطاقة أو خط ائتمان متجدد", "قرض بمبلغ محدد"]


def pct_input(col, label, key, placeholder="0"):
    """نسبة مئوية تُكتب من 0 إلى 100 وتُعاد بين 0 و1."""
    v = col.number_input(f"{label} (%)", min_value=0.0, max_value=100.0, value=None, step=1.0, key=key,
                         placeholder=placeholder)
    return None if v is None else v / 100


def page_lgd_ead():
    st.title("تقييم LGD وEAD")
    st.caption("جميع الحقول إلزامية. المبالغ بالدينار الجزائري (DA).")

    # ---------------- EAD
    st.markdown("### 1. التعرض عند التعثر (EAD)")
    facility = st.selectbox("نوع التسهيل", FACILITY, index=None, key="e_type", placeholder="—")
    ead = None
    if facility == FACILITY[0]:
        a, b, c = st.columns(3)
        limit = money_input(a, "الحد الائتماني", "e_limit")
        drawn = money_input(b, "الرصيد المستعمل", "e_drawn")
        ccf = pct_input(c, "CCF: نسبة استعمال الجزء غير المستعمل", "e_ccf", "مثال: 75")
        missing = missing_fields({"الحد الائتماني": limit, "الرصيد المستعمل": drawn, "CCF": ccf})
        if missing:
            st.info("حقول ناقصة: " + "، ".join(missing) + ".")
        elif limit <= 0:
            st.error("الحد الائتماني يجب أن يكون أكبر من صفر.")
        else:
            ead, used, undrawn = risk.card_ead(limit, drawn, ccf)
            st.markdown(f'<div class="ecl-eq">EAD = {ui.money(used, convert=False)} + {ui.pct(ccf, 0)} × '
                        f'{ui.money(undrawn, convert=False)} = <b>{ui.money(ead, convert=False)}</b></div>',
                        unsafe_allow_html=True)
            st.caption("EAD = الرصيد المستعمل + CCF × الجزء غير المستعمل من الحد.")
    elif facility == FACILITY[1]:
        a, b, c = st.columns(3)
        principal = money_input(a, "مبلغ القرض", "e_principal")
        repaid = money_input(b, "المبلغ المسدَّد", "e_repaid")
        accrued = money_input(c, "الفوائد المستحقة غير المدفوعة", "e_accrued")
        missing = missing_fields({"مبلغ القرض": principal, "المبلغ المسدَّد": repaid, "الفوائد المستحقة": accrued})
        if missing:
            st.info("حقول ناقصة: " + "، ".join(missing) + ".")
        elif repaid > principal:
            st.error("المبلغ المسدَّد أكبر من مبلغ القرض.")
        else:
            ead = risk.loan_ead(principal, repaid, accrued)
            st.markdown(f'<div class="ecl-eq">EAD = {ui.money(principal - repaid, convert=False)} + '
                        f'{ui.money(accrued, convert=False)} = <b>{ui.money(ead, convert=False)}</b></div>',
                        unsafe_allow_html=True)
            st.caption("EAD = المبلغ المتبقي (مبلغ القرض − المسدَّد) + الفوائد المستحقة غير المدفوعة.")
    else:
        st.info("حقول ناقصة: نوع التسهيل.")

    # ---------------- LGD
    st.markdown("### 2. الخسارة عند التعثر (LGD)")
    a, b = st.columns(2)
    collateral = money_input(a, "قيمة الضمان", "l_coll")
    haircut = pct_input(b, "نسبة الاقتطاع من قيمة الضمان", "l_haircut", "مثال: 30")
    a, b = st.columns(2)
    costs = money_input(a, "تكاليف التحصيل", "l_costs")
    unsecured = pct_input(b, "نسبة الاسترداد من الجزء غير المضمون", "l_unsec", "مثال: 10")
    st.caption("الاقتطاع: النسبة التي تُخصم من قيمة الضمان عند بيعه (انخفاض السعر، البيع السريع). "
               "ضمان غير موجود = قيمة 0.")
    missing = missing_fields({"قيمة الضمان": collateral, "نسبة الاقتطاع": haircut, "تكاليف التحصيل": costs,
                              "نسبة الاسترداد من الجزء غير المضمون": unsecured})
    if ead is None:
        st.info("حساب LGD يحتاج EAD من القسم 1.")
        return
    if missing:
        st.info("حقول ناقصة: " + "، ".join(missing) + ".")
        return
    if ead <= 0:
        st.info("EAD يساوي صفرًا: لا يوجد تعرض، فلا خسارة.")
        return
    r = risk.lgd_from_recovery(ead, collateral, haircut, costs, unsecured)
    k = st.columns(4)
    k[0].markdown(ui.kpi("الجزء المغطّى بالضمان", ui.money(r["secured"], convert=False),
                         "قيمة الضمان بعد الاقتطاع، بحد أقصى EAD"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("المبلغ المسترد الصافي", ui.money(r["recovered"], convert=False),
                         "بعد خصم تكاليف التحصيل"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("الخسارة بالمبلغ", ui.money(r["loss"], convert=False), "EAD − المسترد"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("LGD", ui.pct(r["lgd"]), "الخسارة ÷ EAD"), unsafe_allow_html=True)
    st.markdown(f'<div class="ecl-eq">LGD = 1 − {ui.money(r["recovered"], convert=False)} ÷ {ui.money(ead, convert=False)} '
                f'= <b>{ui.pct(r["lgd"])}</b></div>', unsafe_allow_html=True)
    st.caption("المسترد = الجزء المغطّى بالضمان + الجزء غير المضمون × نسبة استرداده − تكاليف التحصيل.")

    # ---------------- EL (اختياري)
    st.markdown("### 3. الخسارة المتوقعة (EL)")
    a, b = st.columns(2)
    pd_ = pct_input(a, "PD: احتمال التعثر", "l_pd", "مثال: 5")
    group = b.selectbox("جدول القرار", ["individuals", "companies"], index=None, key="l_group", placeholder="—",
                        format_func=lambda g: "الأفراد" if g == "individuals" else "الشركات")
    if pd_ is None or group is None:
        st.info("حقول ناقصة: " + "، ".join(n for n, v in (("PD", pd_), ("جدول القرار", group)) if v is None) + ".")
        return
    el = risk.ecl(pd_, r["lgd"], ead)
    decision_block(el, group, pd_, r["lgd"], ead, "من القسم 1")


def main():
    if not st.session_state.get("auth"):
        login_page()
        return
    page = header()
    try:
        with st.spinner("جارٍ تحميل البيانات الحقيقية وتدريب النموذجين (مرة واحدة، قد تستغرق دقيقة)..."):
            ind, comp = get_data()
            res = get_models()
    except dl.DataError as e:
        st.error(f"تعذّر تحميل البيانات: {e}")
        st.stop()
    try:
        if page == "الرئيسية":
            page_home(ind, comp, res)
        elif page == "تقييم فرد":
            page_individual(ind)
        elif page == "تقييم شركة":
            page_company(comp)
        elif page == "تقييم LGD وEAD":
            page_lgd_ead()
        else:
            page_performance(res)
    except ValueError as e:
        st.error(f"قيمة غير صالحة: {e}")
    st.markdown('<p class="small" style="margin-top:2rem">Flow Bank مشروع تعليمي. البيانات من UCI (رخصة CC BY 4.0). '
                'النتائج لا تصلح لقرارات إقراض حقيقية.</p>', unsafe_allow_html=True)


main()
