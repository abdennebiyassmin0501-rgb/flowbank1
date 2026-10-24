"""ui.py — التصميم (CSS) وعناصر واجهة صغيرة قابلة لإعادة الاستعمال."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

INK, TEAL, MINT, SAND = "#0F2027", "#0B5D5A", "#2FB3A3", "#E7EEEC"
RISK_COLORS = {"low": "#1F8A5B", "mid": "#C98A0B", "high": "#B4372C"}
MODEL_COLORS = {"الانحدار اللوجستي": TEAL, "شجرة القرار": "#C98A0B"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');
:root { --ink:#0F2027; --muted:#5B6B70; --line:#DCE5E3; --bg:#F3F6F5; --card:#FFFFFF;
        --teal:#0B5D5A; --mint:#2FB3A3; --mint-soft:#E1F3F0; --amber:#C98A0B; --red:#B4372C; }
/* إخفاء عناصر Streamlit الافتراضية */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
.stDeployButton, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
html, body, .stApp { background: var(--bg); }
.stApp, .stApp * { font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif; }
/* أيقونات Streamlit تُرسم بخط خاص؛ نعيده إليها حتى لا يظهر اسم الأيقونة نصًّا (مثل keyboard_arrow_right) */
.stApp [data-testid="stIconMaterial"], .stApp [class*="material-symbols"], .stApp [class*="material-icons"],
.stApp span[translate="no"] { font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.stApp { direction: rtl; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 3rem !important; max-width: 1180px; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Readex Pro', 'IBM Plex Sans Arabic', sans-serif !important;
  color: var(--ink); text-align: right; letter-spacing: 0; }
.stApp p, .stApp li, .stApp label, .stMarkdown { text-align: right; }
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], [data-testid="stDataFrame"],
[data-testid="stTable"] { direction: ltr; }
[data-testid="stMetricValue"], .num { direction: ltr; unicode-bidi: isolate; font-variant-numeric: tabular-nums; }
/* شريط العلامة */
.fb-bar { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;
  background: var(--ink); color:#EAF4F2; border-radius: 16px; padding: 14px 20px; margin-bottom: 14px; }
.fb-brand { display:flex; align-items:center; gap:12px; }
.fb-logo { width:38px; height:38px; border-radius:11px; background: linear-gradient(135deg, var(--mint), var(--teal));
  position:relative; overflow:hidden; flex:none; }
.fb-logo::after { content:""; position:absolute; inset:9px 6px; border-top:3px solid #fff; border-bottom:3px solid rgba(255,255,255,.55);
  border-radius: 50% 50% 0 0 / 30% 30% 0 0; }
.fb-name { font-family:'Readex Pro',sans-serif; font-weight:700; font-size:1.35rem; direction:ltr; }
.fb-sub { font-size:.82rem; color:#9FC4BE; }
.fb-user { font-size:.85rem; color:#CFE3DF; background: rgba(255,255,255,.08); padding:5px 12px; border-radius:99px; }
/* التنقل */
div[data-testid="stRadio"] > div[role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
.st-key-page div[role="radiogroup"] label { background: var(--card); border:1px solid var(--line); border-radius: 99px;
  padding: 6px 16px !important; margin:0 !important; cursor:pointer; }
.st-key-page div[role="radiogroup"] label:has(input:checked) { background: var(--teal); border-color: var(--teal); }
.st-key-page div[role="radiogroup"] label:has(input:checked) p { color: #fff !important; }
.st-key-page div[role="radiogroup"] label > div:first-of-type { display:none !important; }
.st-key-page div[role="radiogroup"] input { position:absolute; opacity:0; }
/* إخفاء أيقونة الرابط بجانب العناوين */
[data-testid="stHeaderActionElements"], .stApp h1 a[href^="#"], .stApp h2 a[href^="#"], .stApp h3 a[href^="#"] { display:none !important; }
/* بطاقات */
.fb-card { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:18px 20px; height:100%; }
.fb-hero { background: linear-gradient(120deg, #0B5D5A 0%, #0F2027 100%); color:#EAF4F2; border-radius:18px; padding:26px 28px; }
.fb-hero h1 { color:#fff !important; margin:0 0 6px; font-size:1.9rem; }
.fb-hero p { color:#CFE3DF; max-width:70ch; margin:0; }
.fb-kpi { background: var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px; }
.fb-kpi .l { color: var(--muted); font-size:.82rem; }
.fb-kpi .v { font-family:'Readex Pro',sans-serif; font-size:1.6rem; font-weight:600; color: var(--ink); direction:ltr; text-align:right; }
.fb-kpi .s { color: var(--muted); font-size:.78rem; }
.tag { display:inline-block; font-size:.74rem; font-weight:600; padding:2px 10px; border-radius:99px; margin-inline-start:6px; vertical-align:middle; }
.tag.real { background:#E3EEF8; color:#1F5A8A; }
.tag.model { background: var(--mint-soft); color: var(--teal); }
.tag.assume { background:#FBEFD6; color:#8A5D05; }
.tag.input { background:#EEE8F7; color:#5B3E8C; }
.pd-big { font-family:'Readex Pro',sans-serif; font-size:2.4rem; font-weight:700; color: var(--ink); direction:ltr; display:inline-block; }
.pill { display:inline-block; padding:3px 14px; border-radius:99px; font-weight:600; font-size:.9rem; }
.pill.low { background:#DDF1E6; color:#16673F; } .pill.mid { background:#FBEFD6; color:#8A5D05; } .pill.high { background:#F7DEDA; color:#8E2A21; }
.ecl-eq { font-family:'Readex Pro',sans-serif; font-size:1.05rem; direction:ltr; text-align:center; background: var(--bg);
  border-radius:12px; padding:12px; color: var(--ink); }
.cm { border-collapse:collapse; width:100%; font-size:.86rem; }
.cm td, .cm th { border:1px solid var(--line); padding:7px; text-align:center; }
.cm th { background: var(--bg); font-weight:600; }
.cm td.ok { background:#EAF6F0; } .cm td.bad { background:#FBEDEA; }
.small { font-size:.85rem; color: var(--muted); }
/* أزرار ونماذج */
.stButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight:600; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
div[data-testid="stExpander"] { background: var(--card); border-radius: 12px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def tag(kind):
    text = {"real": "بيانات حقيقية", "model": "تنبؤ نموذج مدرَّب", "assume": "افتراض توضيحي", "input": "مُدخَل"}[kind]
    return f'<span class="tag {kind}">{text}</span>'


def kpi(label, value, sub=""):
    return f'<div class="fb-kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


LRI, PDI = "\u2066", "\u2069"   # عزل اتجاه النص: يمنع انقلاب «NT$» إلى «$NT» داخل الجمل العربية


from currency import DA_PER_NTD, RATE_NOTE, to_da, to_ntd  # noqa: F401


def money(v, unit="DA", convert=True):
    """يعرض مبلغًا بالدينار. convert=True: المبلغ بالدولار التايواني فنحوّله؛ False: هو أصلًا بالدينار."""
    if convert:
        v = to_da(v)
    text = f"{float(v):,.0f}".replace(",", " ")
    return f"{LRI}{text} {unit}{PDI}" if unit else f"{LRI}{text}{PDI}"


def pct(x, d=1):
    """نسبة مئوية؛ نزيد الدقة للاحتمالات الصغيرة جدًا حتى لا تظهر 0.2% وهي 0.19%."""
    if d >= 1 and 0 < abs(x) < 0.01:
        d = 2
    return f"{x * 100:.{d}f}%"


def pill(level, names):
    return f'<span class="pill {level}">{names[level]}</span>'


def confusion_html(cm, pos="متعثر", neg="سليم"):
    return (f'<table class="cm"><tr><th></th><th>توقّع: {pos}</th><th>توقّع: {neg}</th></tr>'
            f'<tr><th>فعلًا {pos}</th><td class="ok">{cm["tp"]:,}</td><td class="bad">{cm["fn"]:,}</td></tr>'
            f'<tr><th>فعلًا {neg}</th><td class="bad">{cm["fp"]:,}</td><td class="ok">{cm["tn"]:,}</td></tr></table>')


def _axis(**kw):
    return alt.Axis(labelColor="#5B6B70", titleColor="#5B6B70", gridColor="#E6ECEA", domainColor="#C9D4D1", **kw)


def line_chart(df, x, y, color, x_title, y_title, points=False, height=300):
    scale = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))
    lines = alt.Chart(df).mark_line(strokeWidth=2.5, point=points).encode(
        x=alt.X(f"{x}:Q", title=x_title, axis=_axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title=y_title, axis=_axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(f"{color}:N", scale=scale, title=None, legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip(f"{color}:N", title="النموذج"), alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=".2f")])
    ref = alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]})).mark_line(strokeDash=[5, 5], color="#9AA8A5").encode(x=f"{x}:Q", y=f"{y}:Q")
    st.altair_chart((ref + lines).properties(height=height), use_container_width=True)


def bar_h(df, cat, val, color_field=None, colors=None, title="", fmt=".2f", height=None):
    enc = dict(y=alt.Y(f"{cat}:N", sort=None, title=None, axis=_axis(labelLimit=320)),
               x=alt.X(f"{val}:Q", title=title, axis=_axis()),
               tooltip=[alt.Tooltip(f"{cat}:N", title=" "), alt.Tooltip(f"{val}:Q", format=fmt, title=title)])
    if color_field:
        enc["color"] = alt.Color(f"{color_field}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None)
    mark = alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=TEAL if not color_field else None).encode(**enc)
    st.altair_chart(mark.properties(height=height or 32 * len(df) + 40), use_container_width=True)


def bar_v(df, cat, val, title_x, title_y, fmt=",.0f", color=TEAL, height=260):
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color).encode(
        x=alt.X(f"{cat}:N", sort=None, title=title_x, axis=_axis(labelAngle=0)),
        y=alt.Y(f"{val}:Q", title=title_y, axis=_axis(format=fmt)),
        tooltip=[alt.Tooltip(f"{cat}:N", title=title_x), alt.Tooltip(f"{val}:Q", format=fmt, title=title_y)])
    st.altair_chart(chart.properties(height=height), use_container_width=True)


def fmt_num(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "مفقودة"
    return f"{v:,.{d}f}".replace(",", " ")
