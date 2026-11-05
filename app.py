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
import base64
import html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for font_file in (Path(__file__).parent / "assets" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(font_file))
plt.rcParams["font.family"] = "Product Sans"
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

@st.cache_data(show_spinner=False)
def product_font_css():
    faces = []
    for filename, weight, style in [("regular",400,"normal"),("bold",700,"normal"),("italic",400,"italic"),("bold-italic",700,"italic")]:
        data = base64.b64encode((Path(__file__).parent / "assets" / "fonts" / (filename + ".ttf")).read_bytes()).decode()
        faces.append(f"@font-face{{font-family:'Product Sans';font-style:{style};font-weight:{weight};font-display:swap;src:url(data:font/ttf;base64,{data}) format('truetype');}}")
    return "".join(faces)
st.markdown("<style>" + product_font_css() + Path(__file__).with_name("ui.css").read_text() + "</style>", unsafe_allow_html=True)


NAVY, RED, GREEN, ORANGE, GREY = "#375DFB", "#C25263", "#23876C", "#8798DC", "#778196"
ACCENT = [NAVY, RED, GREEN, ORANGE]
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
                     "font.size": 12})

# Visual thesis: quiet white surfaces, precise type, and a single cobalt action color.
# Content plan: navigation, portfolio metrics, searchable claims, analytical detail.
# Interaction thesis: staggered metrics, responsive navigation, fast hover/focus feedback.
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
# A persistent rail uses the same navigation controls in both sizes.
def download_feedback():
    st.toast("CSV download started.", icon=":material/download:")

if st.session_state.pop("settings_saved", False):
    st.toast("Settings applied.", icon=":material/check:")

if "active_view" not in st.session_state: st.session_state.active_view = "Claims"
# Native checkbox changes layout immediately in the browser; no Python rerun.
st.html('<input type="checkbox" id="claims-navigation" checked aria-label="Show navigation" role="switch">\n<label for="claims-navigation" class="navigation-brand"><span class="nav-shield"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg></span><span class="nav-chevron">›</span><span class="nav-brand-name">ClaimScope</span></label>')
nav = [("Claims", "table_view", "Search, filter, and export claim records"),
       ("Overview", "space_dashboard", "Compare outcomes across policyholder groups"),
       ("Disparities", "query_stats", "Explore differences and income distribution"),
       ("Data preparation", "tune", "Inspect missing values and preview model features"),
       ("Models", "account_tree", "Train and compare four classifiers"),
       ("Evaluation", "monitoring", "Inspect ROC curves and prediction errors"),
       ("Findings", "description", "Review dataset findings and audit next steps")]
for name, icon, detail in nav:
    if st.sidebar.button(name, icon=f":material/{icon}:", key="nav_" + name,
                         type="primary" if st.session_state.active_view == name else "secondary", width="stretch"):
        st.session_state.active_view = name
        st.rerun()
view = st.session_state.active_view

head, actions = st.columns([1.2, 1], gap="small", vertical_alignment="center")
with head:
    st.markdown('<div class="page-title">Insurance claims dashboard</div><div class="page-subtitle">Review outcomes, explore disparities, and evaluate predictive models.</div>', unsafe_allow_html=True)
with actions:
    action_bar = st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="small", key="header_actions")
with action_bar:
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
if "model_config" not in st.session_state:
    st.session_state.model_config = dict(test_size=.25, seed=42, zone_top=15, state_top=15, occ_top=12, reason_top=12)

def close_model_settings():
    st.session_state.settings_open = False

@st.dialog("Model settings", width="medium", on_dismiss=close_model_settings)
def model_settings_dialog():
    cfg = st.session_state.model_config
    st.caption("Configure the training split and feature preparation. Changes apply only when saved.")
    with st.form("model_configuration", border=False):
        st.markdown("**Training**")
        split = st.slider("Holdout for evaluation", .15, .40, cfg["test_size"], .05, help="Fraction reserved for testing. Remaining claims train the models.")
        random_seed = st.number_input("Reproducibility seed", min_value=0, max_value=4294967295, value=cfg["seed"], step=1)
        st.markdown("**Category limits**")
        st.caption("Keep the most frequent categories; combine the rest as Other.")
        left,right = st.columns(2)
        zones = left.number_input("Teams", min_value=5, max_value=30, value=cfg["zone_top"])
        states = right.number_input("States", min_value=5, max_value=30, value=cfg["state_top"])
        occupations = left.number_input("Occupations", min_value=5, max_value=25, value=cfg["occ_top"])
        reasons = right.number_input("Claim reasons", min_value=5, max_value=25, value=cfg["reason_top"])
        if st.form_submit_button("Apply settings", type="primary", icon=":material/check:"):
            st.session_state.model_config = dict(test_size=split, seed=random_seed, zone_top=zones, state_top=states, occ_top=occupations, reason_top=reasons)
            st.session_state.settings_open = False
            st.session_state.settings_saved = True
            st.rerun()

cfg = st.session_state.model_config
test_size, seed, zone_top, state_top, occ_top, reason_top = [cfg[k] for k in ["test_size", "seed", "zone_top", "state_top", "occ_top", "reason_top"]]
with action_bar:
    st.download_button("Export claims", df.to_csv(index=False).encode(), "claims.csv", "text/csv", on_click=download_feedback, type="primary", icon=":material/download:", width="content")
    if st.button("Model settings", icon=":material/tune:", key="open_model_settings", width="content"):
        st.session_state.settings_open = True
if st.session_state.get("settings_open", False):
    model_settings_dialog()
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
            st.dataframe(display, hide_index=True, width="stretch", height=560, row_height=48, key="claims_table",
                         column_config={"Sum assured":st.column_config.NumberColumn(format="localized"), "Age":st.column_config.NumberColumn(format="%d")})
            st.download_button("Export matching claims", filtered.to_csv(index=False).encode(), "filtered_claims.csv", "text/csv", on_click=download_feedback)
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
            ax.text(m*100+0.4, i, f"{m*100:.0f}% (n={n})", va="center", fontsize=12)
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
        st.caption("Use Model settings in the top right to adjust category limits. Preview updates immediately.")
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
                       df.to_csv(index=False).encode(), "insurance_cleaned.csv", "text/csv", on_click=download_feedback)

st.caption("Built with Streamlit + scikit-learn · disparities ≠ proof of bias · "
           "validate against documented repudiation grounds before acting.")
