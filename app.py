"""
Insurance Claim-Settlement Bias & Prediction Dashboard
======================================================
Run locally:        streamlit run app.py
Deploy:             push this folder to GitHub -> share.streamlit.io -> point to app.py

Objectives covered (tabs):
  1. Descriptive analysis  - cross-tabulations against POLICY_STATUS
  2. Diagnostic analysis    - probing repudiation disparity (age / income / team / etc.)
  3. Feature engineering    - cleaning + encoding pipeline
  4. Supervised learning    - KNN, Decision Tree, Random Forest, Gradient Boosting
  5. Evaluation             - accuracy / precision / recall / F1, ROC, confusion matrices
  6. Findings               - written summary with explicit caveats

NOTE ON METHOD: every number is computed from the uploaded data. The app does not
infer intent. A disparity in repudiation rates is a signal to investigate, NOT proof
of bias, because the dataset does not record whether each repudiation was *justified*
(fraud, non-disclosure, policy condition). Caveats are surfaced in-app.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report)

# ----------------------------------------------------------------------------
st.set_page_config(page_title="ClaimScope · Insurance analytics", layout="wide",
                   initial_sidebar_state="expanded")

NAVY, RED, GREEN, ORANGE, GREY = "#375DFB", "#C25263", "#23876C", "#8798DC", "#778196"
ACCENT = [NAVY, RED, GREEN, ORANGE]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.size": 12})

# Visual thesis: quiet white surfaces, precise type, and a single cobalt action color.
# Content plan: navigation, portfolio metrics, searchable claims, analytical detail.
# Interaction thesis: staggered metrics, responsive navigation, fast hover/focus feedback.
st.markdown("""<style>
:root { --unit:4px; --ink:#20232b; --muted:#667085; --line:#e8ebf0; --accent:#375dfb; }
[data-testid="stAppViewContainer"] {background:#f8f9fb;}
[data-testid="stHeader"] {background:rgba(248,249,251,.96);}
[data-testid="stMainBlockContainer"] {padding:80px 32px 48px;max-width:1600px;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid var(--line);width:248px!important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:16px;}
[data-testid="stVerticalBlock"] {gap:16px;}
h1 {font-size:28px!important;letter-spacing:-1px!important;font-weight:650!important;line-height:1.3!important;}
h2 {font-size:20px!important;letter-spacing:-.5px!important;font-weight:600!important;}
h3 {font-size:16px!important;font-weight:600!important;}
p, label {font-size:14px!important;line-height:1.6!important;}
[data-testid="stCaptionContainer"] {color:var(--muted);}
.brand {display:flex;align-items:center;gap:12px;font-size:24px;font-weight:700;letter-spacing:-1px;padding:0 0 20px;border-bottom:1px solid var(--line);}
.brand-mark {background:#20232b;color:white;border-radius:12px;display:grid;place-items:center;width:40px;height:40px;font-size:24px;}
.eyebrow {font-size:12px;color:#697386;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;margin:4px 0 8px;}
.page-title {font-size:24px;font-weight:650;letter-spacing:-.8px;line-height:32px;margin:0;}
.page-subtitle {font-size:13px;color:#667085;margin:4px 0 0;}
.kpi {background:white;border:1px solid var(--line);border-radius:16px;padding:20px 24px;min-height:148px;animation:arrive .32s ease both;transition:border-color .16s,transform .16s,box-shadow .16s;}
.kpi:hover {border-color:#dce2ef;transform:translateY(-2px);box-shadow:0 4px 16px rgba(44,52,72,.07);}
.kpi-label {font-size:13px;color:#667085;margin-bottom:12px;}
.kpi-value {font-size:32px;line-height:40px;font-weight:600;letter-spacing:-1px;font-variant-numeric:tabular-nums;}
.kpi-note {font-size:12px;color:#667085;margin-top:8px;}
.kpi-tag {display:inline-block;background:#edf2ff;color:#3455bc;font-size:12px;border-radius:6px;padding:0 8px;vertical-align:middle;letter-spacing:0;margin-left:8px;}
[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
[data-testid="stBaseButton-segmented_control"],button {transition:background .16s,border-color .16s,box-shadow .16s!important;}
button:hover {border-color:#8ca3ff!important;}
button:focus-visible,a:focus-visible,input:focus-visible {outline:2px solid #375dfb!important;outline-offset:4px!important;}
button:disabled {opacity:.45;cursor:not-allowed!important;}
[data-testid="stExpander"] {background:#fff;border:1px solid var(--line);border-radius:12px;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:8px 12px;border-radius:8px;transition:background .16s;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#f0f3ff;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#edf2ff;color:#375dfb;}
[data-testid="stImage"] {background:white;border-radius:12px;overflow:hidden;}
[data-testid="stMain"] {animation:arrive .24s ease both;}
@keyframes arrive {from {opacity:0;transform:translateY(8px)} to {opacity:1;transform:translateY(0)}}
@media(prefers-reduced-motion:reduce) {*,*::before,*::after {animation:none!important;transition:none!important;scroll-behavior:auto!important;}}
@media(max-width:768px) {[data-testid="stMainBlockContainer"] {padding:72px 16px 32px;} .kpi {padding:16px;min-height:128px;} .kpi-value {font-size:28px;} .page-title {font-size:20px;} }
</style>""", unsafe_allow_html=True)
st.markdown("""<style>
:root {--ink:#34373c;--muted:#75787f;--line:#ececef;--accent:#365cff;}
html,body,[data-testid="stApp"] {color:#34373c;font-family:Arial,Helvetica,sans-serif;}
[data-testid="stAppViewContainer"] {background:#fafafa;}
[data-testid="stHeader"] {background:#fff;border-bottom:1px solid #ededf0;}
[data-testid="stMainBlockContainer"] {padding:88px 24px 40px;}
[data-testid="stVerticalBlock"] {gap:16px;}
[data-testid="stSidebarUserContent"] {padding:16px 12px!important;}
[data-testid="stSidebarHeader"], [data-testid="stSidebarCollapsedControl"] {display:none!important;}
[data-testid="stSidebar"] {transition:width .2s ease,min-width .2s ease;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:8px;}
.rail-brand {display:flex;align-items:center;gap:12px;height:48px;margin:0 4px 16px;font-size:20px;font-weight:600;color:#373a40;white-space:nowrap;}
.brand-mark {width:40px;height:40px;flex-shrink:0;background:#34373c;border-radius:50%;font-weight:400;}
[data-testid="stSidebar"] button {min-height:40px;border:0!important;border-radius:8px;box-shadow:none!important;justify-content:flex-start;padding:8px 12px;color:#73767d;background:transparent;}
[data-testid="stSidebar"] button[kind="primary"] {background:#f0f3ff;color:#365cff;}
[data-testid="stSidebar"] button:hover {background:#f4f5f9;color:#365cff;}
[data-testid="stSidebar"] button:active {background:#e5eaff;transform:scale(.97);}
h1,h2,h3,.page-title {color:#34373c!important;font-weight:500!important;letter-spacing:-.4px!important;}
.page-title {font-size:20px;line-height:28px;}
.page-subtitle {font-size:12px;color:#82858c;}
h2 {font-size:20px!important;} h3 {font-size:18px!important;}
p,label {font-size:13px!important;}
[data-testid="stCaptionContainer"] {color:#85888f;font-size:12px;}
.kpi {border:1px solid #eaeaec;border-radius:16px;padding:24px 20px;box-shadow:0 1px 2px #24272e04;min-height:160px;}
.kpi-value {color:#34373c;font-size:32px;font-weight:500;letter-spacing:-1px;line-height:40px;}
.kpi-label {font-size:13px;color:#74777e;margin-bottom:16px;}
.kpi-note {font-size:12px;color:#92959b;margin-top:8px;}
.kpi-tag {background:#eef2ff;color:#5b6fbc;line-height:20px;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button,[data-testid="stPopover"] button {font-size:13px;border-radius:8px;min-height:40px;}
button:active {transform:translateY(1px);filter:brightness(.96);}
input,[data-baseweb="select"]>div {border-radius:8px!important;}
[data-testid="stDataFrame"] {background:white;border-color:#eeeef0;border-radius:12px;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#eeeef0!important;border-radius:20px!important;background:#fff;}
@media(max-width:768px) {[data-testid="stMainBlockContainer"] {padding:80px 16px 32px;} [data-testid="stSidebar"] {min-width:80px!important;} .kpi {padding:16px;}}
@media(prefers-reduced-motion:reduce) {*,*::before,*::after {transition:none!important;animation:none!important;}}
[data-testid="stApp"] p,[data-testid="stApp"] label,[data-testid="stApp"] small,
[data-testid="stCaptionContainer"], [data-testid="stTooltipContent"],input,button,
[data-baseweb="select"],.eyebrow,.kpi-tag,.kpi-note {font-size: max(12px, 0.75rem)!important;}
[data-testid="stVerticalBlockBorderWrapper"]>div {padding:24px!important;}
</style>""", unsafe_allow_html=True)
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":"#E8EBF0","grid.color":"#E8EBF0",
                     "text.color":"#20232B","axes.labelcolor":"#667085",
                     "xtick.color":"#667085","ytick.color":"#667085",
                     "figure.facecolor":"white","axes.facecolor":"white"})

TARGET = "POLICY_STATUS"
POSITIVE_LABEL = "Repudiate Death"          # positive class = a rejected claim
NUMERIC_COMMA_COLS = ["SUM_ASSURED", "PI_ANNUAL_INCOME"]
ID_COLS = ["POLICY_NO", "PI_NAME"]          # dropped from modelling (identifiers / PII)

# ----------------------------------------------------------------------------
# Data loading + feature engineering (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_engineer(file_bytes: bytes | None):
    if file_bytes is None:
        df = pd.read_csv(Path(__file__).with_name("Insurance.csv"))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    required = ["POLICY_NO", "PI_NAME", "PI_GENDER", "SUM_ASSURED", "ZONE", "PAYMENT_MODE",
                "EARLY_NON", "PI_OCCUPATION", "MEDICAL_NONMED", "PI_STATE", "REASON_FOR_CLAIM",
                "PI_AGE", "PI_ANNUAL_INCOME", "POLICY_STATUS"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("This CSV has no claim records. Add records and upload it again.")
    if len(df) > 100000:
        raise ValueError("Use a CSV with at most 100,000 claims for this dashboard.")
    allowed = {"Approved Death Claim", "Repudiate Death"}
    df[TARGET] = df[TARGET].astype(str).str.strip()
    if not set(df[TARGET]).issubset(allowed):
        raise ValueError("POLICY_STATUS must contain only Approved Death Claim or Repudiate Death.")
    df["PI_AGE"] = pd.to_numeric(df["PI_AGE"], errors="coerce")
    if df["PI_AGE"].isna().any() or not df["PI_AGE"].between(0, 120).all():
        raise ValueError("PI_AGE must contain numbers between 0 and 120.")
    for col in required:
        if col not in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME"]:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    notes = []
    # 1) parse comma-formatted numerics
    for c in NUMERIC_COMMA_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False),
                                  errors="coerce")
    # 2) zero income is almost certainly "not captured" -> flag, do not treat as real 0
    if "PI_ANNUAL_INCOME" in df.columns:
        zero = int((df["PI_ANNUAL_INCOME"] == 0).sum())
        df["INCOME_MISSING"] = (df["PI_ANNUAL_INCOME"] == 0).astype(int)
        notes.append(f"PI_ANNUAL_INCOME = 0 for {zero} rows ({zero/len(df)*100:.1f}%) — "
                     f"treated as MISSING (flag added), not as genuine zero income.")
    # 3) normalise ZONE case variants (South / SOUTH / south ...)
    if "ZONE" in df.columns:
        before = df["ZONE"].nunique()
        df["ZONE"] = df["ZONE"].astype(str).str.strip().str.upper()
        notes.append(f"ZONE case-normalised: {before} -> {df['ZONE'].nunique()} distinct teams.")
    # 4) explicit Unknown for missing categoricals
    for c in ["REASON_FOR_CLAIM", "PI_OCCUPATION"]:
        if c in df.columns:
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna("Unknown")
            if n: notes.append(f"{c}: {n} missing -> 'Unknown'.")
    # 5) binary target
    df["REPUDIATED"] = (df[TARGET] == POSITIVE_LABEL).astype(int)
    return df, notes


def bucket_rare(s: pd.Series, top_n: int, other="Other"):
    keep = s.value_counts().head(top_n).index
    return np.where(s.isin(keep), s, other)


@st.cache_data(show_spinner=False)
def build_design_matrix(df: pd.DataFrame, zone_top, state_top, occ_top, reason_top):
    d = df.copy()
    for col, t in [("ZONE", zone_top), ("PI_STATE", state_top),
                   ("PI_OCCUPATION", occ_top), ("REASON_FOR_CLAIM", reason_top)]:
        if col in d.columns:
            d[col] = bucket_rare(d[col], t)
    num = [c for c in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME", "INCOME_MISSING"] if c in d]
    cat = [c for c in ["PI_GENDER", "ZONE", "PAYMENT_MODE", "EARLY_NON", "MEDICAL_NONMED",
                       "PI_OCCUPATION", "PI_STATE", "REASON_FOR_CLAIM"] if c in d]
    X = d[num + cat]
    y = d["REPUDIATED"]
    return X, y, num, cat


def make_preprocessor(num, cat):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


@st.cache_resource(show_spinner=True)
def train_models(_X, _y, num, cat, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                          stratify=_y, random_state=seed)
    pre = make_preprocessor(num, cat)
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte, target_names=["Approved", "Repudiated"],
                                             zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


def cramers_v(df, col, target="REPUDIATED"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.markdown("""<style>
/* One brand control, one aligned icon column, no selected-state boxes. */
[data-testid="stSidebar"] button {justify-content:flex-start!important;padding:8px 12px!important;min-height:40px!important;}
[data-testid="stSidebar"] button[kind="primary"] {background:transparent!important;color:#365cff!important;}
[data-testid="stSidebar"] button:hover {background:transparent!important;color:#34373c!important;}
[data-testid="stSidebar"] button:active {background:transparent!important;color:#26292f!important;transform:translateY(1px);}
.st-key-toggle_rail {margin-bottom:16px;}
.st-key-toggle_rail button {height:48px!important;color:#45484f!important;}
.st-key-toggle_rail button [data-testid="stIconMaterial"] {background:#373a40;color:white;border-radius:50%;padding:8px;font-size:24px!important;min-width:40px;height:40px;}
.st-key-toggle_rail button:hover [data-testid="stIconMaterial"] {font-size:0!important;}
.st-key-toggle_rail button:hover [data-testid="stIconMaterial"]::after {content:'›';font-family:Arial,sans-serif;font-size:28px;line-height:24px;}
[data-testid="stButton"] button:hover,[data-testid="stDownloadButton"] button:hover,[data-testid="stPopover"] button:hover {border-color:#b3b6be!important;filter:brightness(.94);box-shadow:none!important;}
[data-testid="stSidebar"] button:hover {filter:none!important;}
[data-testid="stMainBlockContainer"] {padding-top:64px!important;}
.kpi {margin-bottom:16px;}
</style>""", unsafe_allow_html=True)
st.markdown("""<style>
:root {--space:4px;--radius:8px;}
[data-testid="stSidebarUserContent"] {padding:12px 8px!important;overflow:hidden;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:4px!important;}
[data-testid="stSidebar"] button {height:32px!important;min-height:32px!important;padding:4px 12px!important;gap:12px!important;border-radius:8px!important;white-space:nowrap;}
[data-testid="stSidebar"] button p {font-size:13px!important;white-space:nowrap;}
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {font-size:20px!important;min-width:20px;}
.st-key-toggle_rail {margin-bottom:8px!important;}
.st-key-toggle_rail button {height:40px!important;min-height:40px!important;}
.st-key-toggle_rail button [data-testid="stIconMaterial"] {min-width:24px!important;width:24px!important;height:24px!important;padding:0!important;border-radius:8px!important;background:transparent!important;color:#484b52!important;}
.st-key-toggle_rail button:hover [data-testid="stIconMaterial"] {font-size:0!important;}
.st-key-toggle_rail button:hover [data-testid="stIconMaterial"]::after {font-size:24px!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {padding:8px 12px;}
[data-testid="stSidebar"] [data-testid="stPopoverButton"] {margin-top:8px;}
[data-testid="stMainBlockContainer"] {padding:64px 24px 32px!important;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap:16px!important;}
.kpi {margin:0!important;min-height:144px!important;padding:20px!important;border-radius:8px!important;}
.kpi-label {margin-bottom:12px!important;}
.kpi-note {margin-top:8px!important;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:8px!important;padding:20px!important;background:#fff!important;}
[data-testid="stMainBlockContainer"] button {min-height:36px!important;height:36px!important;padding:4px 12px!important;white-space:nowrap;}
button,input,[data-baseweb="select"]>div,[data-testid="stExpander"],[data-testid="stDataFrame"],[data-testid="stAlert"],[data-testid="stPopoverBody"] {border-radius:8px!important;}
[data-testid="stPopoverBody"] {min-width:320px;max-width:min(400px,calc(100vw - 32px));}
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {gap:12px!important;}
@media(max-width:768px) {[data-testid="stMainBlockContainer"] {padding:64px 16px 24px!important;} [data-testid="stSidebar"]:hover {box-shadow:none!important;}}
</style>""", unsafe_allow_html=True)
# A persistent rail uses the same navigation controls in both sizes.
if "rail_compact" not in st.session_state: st.session_state.rail_compact = True
if "active_view" not in st.session_state: st.session_state.active_view = "Claims"
compact = st.session_state.rail_compact
rail_width = 64 if compact else 240
st.markdown("<style>[data-testid=stSidebar] {width:%dpx!important;min-width:%dpx!important;max-width:%dpx!important;}" % (rail_width, rail_width, rail_width) + ('[data-testid="stSidebar"]:hover,[data-testid="stSidebar"]:focus-within {width:240px!important;min-width:240px!important;max-width:240px!important;margin-right:-176px;z-index:1000;box-shadow:4px 0 16px #34373c0d;}\n[data-testid="stSidebar"]:not(:hover):not(:focus-within) button p {display:none;}\n[data-testid="stSidebar"]:not(:hover):not(:focus-within) [data-testid="stPopoverButton"]>svg {display:none;}' if compact else "") + "</style>", unsafe_allow_html=True)
if st.sidebar.button("ClaimScope", icon=":material/shield:", help="Pin navigation open" if compact else "Unpin navigation", key="toggle_rail", width="stretch"):
    st.session_state.rail_compact = not compact
    st.rerun()
nav = [("Claims", "table_view", "Search, filter, and export claim records"),
       ("Overview", "space_dashboard", "Compare outcomes across policyholder groups"),
       ("Disparities", "query_stats", "Explore differences and income distribution"),
       ("Data preparation", "tune", "Inspect missing values and preview model features"),
       ("Models", "account_tree", "Train and compare four classifiers"),
       ("Evaluation", "monitoring", "Inspect ROC curves and prediction errors"),
       ("Findings", "description", "Review dataset findings and audit next steps")]
for name, icon, detail in nav:
    if st.sidebar.button(name, icon=f":material/{icon}:", help=f"{name} — {detail}", key="nav_" + name,
                         type="primary" if st.session_state.active_view == name else "secondary", width="stretch"):
        st.session_state.active_view = name
        st.rerun()
view = st.session_state.active_view
head, upload_action, export_action = st.columns([6, 1.15, 1.15], gap="small", vertical_alignment="center")
with head:
    st.markdown('<div class="page-title">Insurance claims dashboard</div><div class="page-subtitle">Review outcomes, explore disparities, and evaluate predictive models.</div>', unsafe_allow_html=True)
with upload_action:
    with st.popover("Upload claims", icon=":material/upload_file:", width="content"):
        up = st.file_uploader("Upload claims CSV", type=["csv"], help="Use the 14-column sample schema. Maximum 10 MB.")
        st.download_button("Download CSV template", Path(__file__).with_name("Insurance.csv").read_text().splitlines()[0] + "\n", "claims_template.csv", "text/csv", width="stretch")
file_bytes = up.getvalue() if up is not None else None
try:
    if file_bytes is not None and len(file_bytes) > 10*1024*1024:
        raise ValueError("File exceeds 10 MB. Upload a smaller CSV.")
    df, notes = load_and_engineer(file_bytes)
except (ValueError, OSError, UnicodeError, pd.errors.ParserError) as exc:
    st.title("Unable to load claims")
    st.error(str(exc))
    st.info("Open Upload claims above. Replace the CSV, or remove it to return to the bundled sample.")
    st.stop()
with st.sidebar.popover("Model settings", icon=":material/settings:", help="Configure training and feature preparation", width="stretch"):
    st.subheader("Model settings")
    st.caption("Changes apply when you save. Models use the complete loaded dataset.")
    with st.form("model_configuration", border=False):
        st.markdown("**Training**")
        test_size = st.slider("Holdout for evaluation", 0.15, 0.40, 0.25, 0.05, format="%.2f", help="Fraction reserved for testing; remaining claims train the models.")
        seed = st.number_input("Reproducibility seed", value=42, min_value=0, max_value=4294967295, step=1)
        st.markdown("**Category limits**")
        st.caption("Keep the most frequent categories; combine the rest as Other.")
        left, right = st.columns(2)
        zone_top = left.number_input("Teams", min_value=5, max_value=30, value=15, step=1)
        state_top = right.number_input("States", min_value=5, max_value=30, value=15, step=1)
        occ_top = left.number_input("Occupations", min_value=5, max_value=25, value=12, step=1)
        reason_top = right.number_input("Claim reasons", min_value=5, max_value=25, value=12, step=1)
        saved = st.form_submit_button("Apply settings", type="primary", icon=":material/check:")
    if saved: st.success("Settings applied.")
if not compact:
    st.sidebar.caption("Yudha’s workspace · ClaimScope")
    st.sidebar.caption("Disparities are audit leads, not proof of bias.")
with export_action:
    st.download_button("Export claims", df.to_csv(index=False).encode(), "claims.csv", "text/csv", type="primary", icon=":material/download:", width="content")
overall = df["REPUDIATED"].mean()
for i, (col, label, value, tag, note) in enumerate(zip(st.columns(4),
    ["Total claims", "Approved claims", "Repudiated claims", "Income recorded"],
    [f"{len(df):,}", f"{(df.REPUDIATED==0).sum():,}", f"{df.REPUDIATED.sum():,}", f"{(df.PI_ANNUAL_INCOME.fillna(0)>0).mean()*100:.1f}%"],
    ["All records", f"{(1-overall)*100:.1f}%", f"{overall*100:.1f}%", "Data quality"],
    ["Current dataset", "Share of all claims", "Share of all claims", "Non-zero annual income"])):
    col.markdown(f'<div class="kpi" style="animation-delay:{i*40}ms"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}<span class="kpi-tag">{tag}</span></div></div>', unsafe_allow_html=True)
if view == "Claims":
    with st.container(border=True):
        st.subheader("Claim records", icon=":material/table_view:")
        st.caption("Find a policyholder or policy, narrow the list, then export the matching claims.")
        search_col, status_col, state_col = st.columns([2, 1, 1])
        query = search_col.text_input("Search claims", placeholder="Name or policy number…")
        status = status_col.selectbox("Outcome", ["All outcomes", "Approved Death Claim", "Repudiate Death"])
        state = state_col.selectbox("State", ["All states"] + sorted(df.PI_STATE.unique()))
        filtered = df.copy()
        if query.strip():
            filtered = filtered[filtered.PI_NAME.str.contains(query.strip(), case=False, regex=False) | filtered.POLICY_NO.astype(str).str.contains(query.strip(), case=False, regex=False)]
        if status != "All outcomes":
            filtered = filtered[filtered.POLICY_STATUS == status]
        if state != "All states":
            filtered = filtered[filtered.PI_STATE == state]
        st.caption(f"{len(filtered):,} of {len(df):,} claims")
        if filtered.empty:
            st.info("No claims match these filters. Clear the search or choose All outcomes and All states.")
        else:
            display = filtered[["PI_NAME", "POLICY_NO", "PI_STATE", "PI_AGE", "SUM_ASSURED", "POLICY_STATUS"]].rename(columns={"PI_NAME":"Policyholder", "POLICY_NO":"Policy number", "PI_STATE":"State", "PI_AGE":"Age", "SUM_ASSURED":"Sum assured", "POLICY_STATUS":"Outcome"})
            st.dataframe(display, hide_index=True, width="stretch", height=560, row_height=48,
                         column_config={"Sum assured":st.column_config.NumberColumn(format="localized"), "Age":st.column_config.NumberColumn(format="%d")})
            st.download_button("Export matching claims", filtered.to_csv(index=False).encode(), "filtered_claims.csv", "text/csv")
        with st.expander("Data quality notes"):
            for n in notes: st.write(n)

# ============================== TAB 1 — DESCRIPTIVE =========================
if view == "Overview":
    st.header("Descriptive Analysis — Cross-Tabulation vs POLICY_STATUS")
    st.caption("Each row of a cross-tab is row-normalised to show the **repudiation rate** "
               "within that group. Dashed line = overall rate.")

    cat_options = [c for c in ["PI_GENDER", "EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE",
                               "ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]
                   if c in df.columns]
    sel = st.selectbox("Cross-tabulate POLICY_STATUS against:", cat_options, index=0)

    colL, colR = st.columns([1, 1])
    with colL:
        st.subheader(f"Counts: {sel} × POLICY_STATUS")
        ct_counts = pd.crosstab(df[sel], df[TARGET], margins=True, margins_name="Total")
        st.dataframe(ct_counts, width='stretch')
    with colR:
        st.subheader("Row % (repudiation rate by group)")
        ct_pct = (pd.crosstab(df[sel], df[TARGET], normalize="index") * 100).round(1)
        st.dataframe(ct_pct, width='stretch')

    # chart
    g = df.groupby(sel)["REPUDIATED"].agg(["mean", "count"])
    g = g[g["count"] >= 5].sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(g))))
    colors = [RED if m > overall else GREEN for m in g["mean"]]
    ax.barh(g.index.astype(str), g["mean"] * 100, color=colors)
    ax.axvline(overall * 100, color=NAVY, ls="--", lw=1.4, label=f"Overall {overall*100:.0f}%")
    for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
        ax.text(m * 100 + 0.4, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=12)
    ax.set_xlabel("Repudiation %"); ax.legend(); ax.set_title(f"Repudiation rate by {sel}")
    st.pyplot(fig, width='stretch')

    st.subheader("Numeric summary by outcome")
    num_cols = [c for c in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME"] if c in df.columns]
    st.dataframe(df.groupby(TARGET)[num_cols].describe().T, width='stretch')

# ============================== TAB 2 — DIAGNOSTIC =========================
if view == "Disparities":
    st.header("Diagnostic Analysis — Probing Differential Treatment")
    st.warning("**Interpretation guardrail:** a higher repudiation rate for a group is a "
               "*disparity*, not automatically *bias*. The data does not record whether each "
               "repudiation was justified (fraud / non-disclosure / policy condition). "
               "Treat large gaps as **leads to audit**, confirm with claim-file review.")

    # association strength ranking
    st.subheader("Which factors are most associated with the outcome?")
    st.caption("χ² test of independence + Cramér's V effect size. Higher V = stronger "
               "association. p ≥ 0.05 (grey) = not statistically significant.")
    assoc_cols = [c for c in ["ZONE", "PI_STATE", "PAYMENT_MODE", "PI_OCCUPATION",
                              "EARLY_NON", "MEDICAL_NONMED", "PI_GENDER"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Factor": c, "Cramér's V": round(v, 3), "χ²": round(chi2, 1),
                     "dof": dof, "p-value": f"{p:.2e}", "categories": ncat,
                     "significant (p<0.05)": "✅" if p < 0.05 else "—"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
    st.dataframe(assoc, width='stretch', hide_index=True)
    st.caption("⚠️ High-cardinality factors (many categories with few rows each) can inflate "
               "χ²/V and violate the test's small-cell assumptions — read ZONE/STATE/"
               "PAYMENT_MODE as the robust, interpretable drivers.")

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    bar_colors = [GREY if "—" in s else NAVY for s in a2["significant (p<0.05)"]]
    ax.barh(a2["Factor"], a2["Cramér's V"], color=bar_colors)
    for i, v in enumerate(a2["Cramér's V"]):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=12)
    ax.set_xlabel("Cramér's V (effect size)")
    st.pyplot(fig, width='stretch')

    colA, colB = st.columns(2)

    # team / zone
    with colA:
        st.subheader("By TEAM / ZONE")
        minn = st.slider("Min claims per team", 10, 60, 25, key="zmin")
        g = df.groupby("ZONE")["REPUDIATED"].agg(["mean", "count"])
        g = g[g["count"] >= minn].sort_values("mean")
        fig, ax = plt.subplots(figsize=(7, max(3, 0.34 * len(g))))
        ax.barh(g.index, g["mean"] * 100, color=[RED if m > overall else GREEN for m in g["mean"]])
        ax.axvline(overall * 100, color=NAVY, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(m*100+0.4, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=7.5)
        ax.set_xlabel("Repudiation %")
        st.pyplot(fig, width='stretch')
        if len(g) >= 2:
            hi, lo = g.iloc[-1], g.iloc[0]
            st.error(f"Spread: **{g.index[-1]} {hi['mean']*100:.0f}%** vs "
                     f"**{g.index[0]} {lo['mean']*100:.0f}%** — "
                     f"{hi['mean']/max(lo['mean'],1e-9):.1f}× difference.")

    # age
    with colB:
        st.subheader("By AGE BAND")
        bins = [0, 18, 30, 40, 50, 60, 70, 200]
        labs = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        ab = pd.cut(df["PI_AGE"], bins=bins, labels=labs, right=False)
        g = df.assign(AB=ab).groupby("AB", observed=True)["REPUDIATED"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(str), g["mean"] * 100, color=NAVY)
        ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=12)
        ax.set_ylabel("Repudiation %")
        st.pyplot(fig, width='stretch')

        st.subheader("By INCOME QUARTILE")
        income_min = st.number_input("Minimum recorded annual income", min_value=0, value=0, step=10000, help="Explore income groups above this threshold. Model training still uses the complete dataset.")
        dz = df[df["PI_ANNUAL_INCOME"] > income_min].copy()
        st.caption(f"Only **{len(dz)}** of {len(df)} rows have income > 0 "
                   f"(rest are 0/missing) — read with caution.")
        if len(dz) > 20:
            dz["Q"] = pd.qcut(dz["PI_ANNUAL_INCOME"], 4, duplicates="drop")
            g = dz.groupby("Q", observed=True)["REPUDIATED"].agg(["mean", "count"])
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(g.index.astype(str), g["mean"] * 100, color=ORANGE)
            ax.axhline(overall * 100, color=RED, ls="--", lw=1.3)
            for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
                ax.text(i, m*100+0.6, f"{m*100:.0f}%\nn={n}", ha="center", fontsize=12)
            ax.set_ylabel("Repudiation %")
            st.pyplot(fig, width='stretch')

# ============================== TAB 3 — FEATURE ENGINEERING ================
if view == "Data preparation":
    st.header("Data preparation")
    st.caption("Inspect a field, adjust category limits, and preview exactly what enters the model.")
    detail, preview = st.tabs(["Field quality", "Model features"])
    with detail:
        field = st.selectbox("Inspect field", ["PI_ANNUAL_INCOME", "PI_AGE", "SUM_ASSURED", "ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"])
        raw = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes is not None else pd.read_csv(Path(__file__).with_name("Insurance.csv"))
        m1,m2,m3 = st.columns(3)
        m1.metric("Records", f"{len(raw):,}")
        m2.metric("Missing values", f"{raw[field].isna().sum():,}")
        m3.metric("Distinct values", f"{raw[field].nunique():,}")
        if field == "PI_ANNUAL_INCOME":
            st.info("Zero income means not recorded. The model flags these records and imputes income from the training set median.")
            st.bar_chart(pd.Series({"Recorded":int((df[field]>0).sum()), "Missing or zero":int((df[field].fillna(0)<=0).sum())}), color=NAVY)
        elif field in ["PI_AGE", "SUM_ASSURED"]:
            st.bar_chart(df[field].value_counts(bins=12).sort_index().rename_axis("Range").reset_index(name="Claims").assign(Range=lambda x:x.Range.astype(str)).set_index("Range"), color=NAVY)
        else:
            limit = st.slider("Categories to preview", 4, 24, 12, step=4)
            st.bar_chart(df[field].value_counts().head(limit), color=NAVY)
        comparison = pd.DataFrame({"Original value":raw[field].astype(str).head(20), "Prepared value":df[field].astype(str).head(20)})
        st.dataframe(comparison, hide_index=True, width="stretch")
    with preview:
        st.caption("Use Model settings in the navigation to adjust category limits. Preview updates immediately.")
        X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
        selected = st.multiselect("Preview columns", list(X.columns), default=list(X.columns)[:6])
        if selected:
            st.dataframe(X[selected].head(20), width="stretch", hide_index=True)
        else:
            st.info("Choose at least one column to preview model features.")
        pre = make_preprocessor(num, cat).fit(X)
        st.caption(f"{len(X):,} rows · {pre.transform(X).shape[1]} encoded features · identifiers excluded")
        with st.expander("Transformation details"):
            for note in notes: st.write(note)
            st.write("Numeric values are median-imputed and standardized. Categories are one-hot encoded; unseen categories are handled safely.")

# ============================== TAB 4 — MODELS =============================
if view in ["Models", "Evaluation"] and (len(df) < 24 or df.REPUDIATED.value_counts().min() < 6 or df.REPUDIATED.nunique() < 2):
    st.warning("Models need at least 24 records and 6 claims of each outcome. Use a larger dataset or return to Claims.")
    st.stop()

if view == "Models":
    st.header("Supervised Classification Models")
    st.caption("Models: KNN (k=15), Decision Tree (depth 6, balanced), "
               "Random Forest (300 trees, depth 12, balanced), Gradient Boosting (defaults). "
               "Trees use `class_weight='balanced'` to offset the 68/32 class imbalance.")

    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    results, (ntr, nte, tr_pos, te_pos) = train_models(X, y, num, cat, test_size, int(seed))
    st.write(f"Train rows: **{ntr:,}** (repud {tr_pos*100:.1f}%) | "
             f"Test rows: **{nte:,}** (repud {te_pos*100:.1f}%)")

    metrics = pd.DataFrame({
        m: {"Train Acc": r["train_acc"], "Test Acc": r["test_acc"],
            "Precision": r["precision"], "Recall": r["recall"],
            "F1": r["f1"], "ROC-AUC": r["roc_auc"],
            "Overfit gap": r["train_acc"] - r["test_acc"]}
        for m, r in results.items()
    }).T.round(3)
    st.subheader("Metric comparison (test set)")
    st.dataframe(metrics.style.highlight_max(axis=0, subset=["Test Acc", "Precision",
                 "Recall", "F1", "ROC-AUC"], color="#d4efdf")
                 .highlight_max(axis=0, subset=["Overfit gap"], color="#f5b7b1"),
                 width='stretch')

    best = metrics["ROC-AUC"].idxmax()
    st.success(f"🏆 Best ROC-AUC: **{best}** ({metrics.loc[best,'ROC-AUC']:.3f}). "
               f"Random Forest also tends to have the largest train–test gap → strongest "
               f"but most prone to overfitting; Decision Tree generalises more tightly.")

    st.subheader("Per-model classification report")
    pick = st.selectbox("Model", list(results.keys()), index=2)
    st.code(results[pick]["report"])
    st.session_state["results"] = results   # share with eval tab

# ============================== TAB 5 — EVALUATION =========================
if view == "Evaluation":
    st.header("Model Evaluation — Stability & Errors")
    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    results, _ = train_models(X, y, num, cat, test_size, int(seed))

    st.subheader("ROC curves")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for (name, r), c in zip(results.items(), ACCENT):
        fpr, tpr, _ = r["roc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig, width='stretch')
    st.caption("AUC well above 0.5 ⇒ there is learnable structure: the outcome is "
               "*predictable* from policy attributes. Combined with Tab 2, much of that "
               "structure is the processing channel (ZONE/STATE).")

    st.subheader("Metric comparison")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mets = ["test_acc", "precision", "recall", "f1", "roc_auc"]
    labels = ["Test Acc", "Precision", "Recall", "F1", "ROC-AUC"]
    xx = np.arange(len(labels)); w = 0.2
    for i, ((name, r), c) in enumerate(zip(results.items(), ACCENT)):
        ax.bar(xx + i*w - 1.5*w, [r[m] for m in mets], w, label=name, color=c)
    ax.set_xticks(xx); ax.set_xticklabels(labels); ax.set_ylim(0, 1); ax.legend(ncol=2, fontsize=12)
    st.pyplot(fig, width='stretch')

    st.subheader("Confusion matrices (test set)")
    cols = st.columns(2)
    for i, (name, r) in enumerate(results.items()):
        with cols[i % 2]:
            cm = r["cm"]
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(cm, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontweight="bold",
                            color="white" if cm[a, b] > cm.max()/2 else NAVY, fontsize=13)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred Approve", "Pred Repud"])
            ax.set_yticklabels(["Act Approve", "Act Repud"]); ax.grid(False)
            ax.set_title(f"{name}\nacc={r['test_acc']:.2f} · recall={r['recall']:.2f}",
                         fontsize=12)
            st.pyplot(fig, width='stretch')

    # feature importance (RF)
    st.subheader("Feature importance (Random Forest)")
    rf = results["Random Forest"]["pipe"]
    ohe = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    feat = num + list(ohe.get_feature_names_out(cat))
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color=GREEN)
    ax.set_xlabel("Importance")
    st.pyplot(fig, width='stretch')
    st.caption("One-hot encoding splits each categorical across many dummies, so the "
               "*aggregate* importance of ZONE/STATE is understated here — sum the dummies "
               "to gauge a factor's true weight.")

# ============================== TAB 6 — FINDINGS ===========================
if view == "Findings":
    st.header("Findings & Recommendations")
    st.write(f"Of {len(df):,} claims, {int(df.REPUDIATED.sum()):,} were repudiated ({overall:.1%}).")
    group = df.groupby("ZONE").REPUDIATED.agg(["mean", "count"])
    eligible = group[group["count"] >= 20].sort_values("mean")
    if len(eligible) >= 2:
        st.write(f"Among teams with at least 20 claims, repudiation ranges from {eligible['mean'].min():.1%} to {eligible['mean'].max():.1%}. Compare case mix and documented grounds before interpreting this gap.")
    else:
        st.info("Too few teams have 20 claims for a stable team comparison.")
    st.write(f"Annual income is missing or zero for {(df.PI_ANNUAL_INCOME.fillna(0) <= 0).mean():.1%} of records. Treat income comparisons cautiously.")
    st.subheader("Audit next steps")
    st.markdown("""1. Review a stratified sample from teams with high and low repudiation rates.
2. Verify documented policy grounds and case mix with an independent reviewer.
3. Improve missing income capture before drawing income-related conclusions.
4. Compare models with and without geography and processing-channel attributes.""")
    st.info("Disparities are leads for audit, not proof of bias. This dataset does not establish whether each repudiation was justified.")
    st.download_button("⬇️ Download cleaned dataset (CSV)",
                       df.to_csv(index=False).encode(), "insurance_cleaned.csv", "text/csv")

st.caption("Built with Streamlit + scikit-learn · disparities ≠ proof of bias · "
           "validate against documented repudiation grounds before acting.")
