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
import altair as alt
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


# Colours below pass the colourblind-separation checks on a white surface.
NAVY, RED, GREY = "#375DFB", "#C25263", "#778196"
MODEL_COLORS = ["#375DFB", "#EB6834", "#1BAF7A", "#6B3FA0"]
INK, AXIS_TEXT, LINE = "#34373C", "#5F636B", "#E9EAED"

# Visual thesis: quiet white surfaces, precise type, and a single cobalt action color.
# Content plan: navigation, portfolio metrics, searchable claims, analytical detail.
# Interaction thesis: staggered metrics, responsive navigation, fast hover/focus feedback.

def show_chart(chart):
    """Draw a chart as SVG, so it stays sharp at any size and redraws from live data."""
    chart = (chart.properties(usermeta={"embedOptions": {"renderer": "svg"}})
             .configure(font="Product Sans", background="white", padding=8)
             .configure_view(stroke=None)
             .configure_axis(labelFontSize=12, labelColor=AXIS_TEXT, labelLimit=240,
                             titleFontSize=12, titleFontWeight="normal", titleColor=AXIS_TEXT,
                             gridColor=LINE, domainColor=LINE, ticks=False, tickCount=6, labelPadding=6)
             .configure_legend(orient="top", title=None, labelFontSize=12, labelColor=INK,
                               symbolType="square", labelLimit=320)
             .configure_title(anchor="start", fontSize=14, fontWeight="normal", color=INK,
                              subtitleFontSize=12, subtitleColor=AXIS_TEXT))
    st.altair_chart(chart, width="stretch", theme=None)

def interval_label(interval):
    """Readable range for a pandas Interval, e.g. 45,000–90,000."""
    return f"{max(interval.left, 0):,.0f}–{interval.right:,.0f}"

def rate_chart(g, overall, horizontal=True):
    """Repudiation rate per group from a frame with mean/count, in display order, against the overall rate."""
    if g.empty:
        st.info("No groups meet this threshold. Lower it to see results.")
        return
    above, below = f"Above overall ({overall*100:.0f}%)", f"At or below overall ({overall*100:.0f}%)"
    data = pd.DataFrame({"Group": g.index.astype(str), "Rate": (g["mean"] * 100).to_numpy(),
                         "Claims": g["count"].to_numpy()})
    data["Position"] = np.where(g["mean"] > overall, above, below)
    data["Label"] = [f"{m:.0f}% (n={n})" for m, n in zip(data.Rate, data.Claims)]
    data["RateLabel"] = [f"{m:.0f}%" for m in data.Rate]
    data["CountLabel"] = [f"n={n}" for n in data.Claims]
    domain = [0, max(data.Rate.max(), overall * 100) * 1.3]
    rate = alt.Scale(domain=domain, nice=False)
    rate_axis = alt.Axis(title="Repudiation %", grid=True)
    tooltip = [alt.Tooltip("Group:N"), alt.Tooltip("Rate:Q", title="Repudiation %", format=".1f"),
               alt.Tooltip("Claims:Q", format=",")]
    color = alt.Color("Position:N", scale=alt.Scale(domain=[above, below], range=[RED, NAVY]))
    base = alt.Chart(data)
    reference = alt.Chart(pd.DataFrame({"Overall": [overall * 100]})).mark_rule(
        color=INK, strokeDash=[4, 4], strokeWidth=1.5)
    if horizontal:
        order = data.Group.tolist()[::-1]
        group = alt.Y("Group:N", sort=order, title=None, axis=alt.Axis(grid=False))
        value = alt.X("Rate:Q", scale=rate, axis=rate_axis)
        bars = base.mark_bar(cornerRadiusEnd=4, height={"band": .72}).encode(y=group, x=value, color=color, tooltip=tooltip)
        labels = base.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(y=group, x=value, text="Label:N")
        line = reference.encode(x=alt.X("Overall:Q", scale=rate), tooltip=alt.Tooltip("Overall:Q", format=".1f"))
        height = 30 * len(data) + 70
    else:
        group = alt.X("Group:N", sort=data.Group.tolist(), title=None, axis=alt.Axis(grid=False, labelAngle=0))
        value = alt.Y("Rate:Q", scale=rate, axis=rate_axis)
        bars = base.mark_bar(cornerRadiusEnd=4, width={"band": .72}).encode(x=group, y=value, color=color, tooltip=tooltip)
        labels = (base.mark_text(baseline="bottom", dy=-18, color=INK, fontSize=12).encode(x=group, y=value, text="RateLabel:N")
                  + base.mark_text(baseline="bottom", dy=-4, color=AXIS_TEXT, fontSize=12).encode(x=group, y=value, text="CountLabel:N"))
        line = reference.encode(y=alt.Y("Overall:Q", scale=rate), tooltip=alt.Tooltip("Overall:Q", format=".1f"))
        height = 320
    show_chart((bars + line + labels).properties(height=height))

def value_bars(labels, values, value_title, fmt=",.0f", groups=None):
    """Ranked horizontal bars, drawn top-down in the order given; groups maps group name to colour."""
    data = pd.DataFrame({"Item": [str(x) for x in labels], "Value": [float(x) for x in values]})
    if data.empty:
        st.info("Nothing to chart for this selection.")
        return
    item = alt.Y("Item:N", sort=data.Item.tolist(), title=None, axis=alt.Axis(grid=False))
    value = alt.X("Value:Q", title=value_title, scale=alt.Scale(domain=[0, max(data.Value.max(), 1e-9) * 1.18], nice=False))
    tooltip = [alt.Tooltip("Item:N"), alt.Tooltip("Value:Q", title=value_title, format=fmt)]
    if groups:
        data["Group"] = groups["assign"]
        color = alt.Color("Group:N", scale=alt.Scale(domain=list(groups["colors"]), range=list(groups["colors"].values())))
    else:
        color = alt.value(NAVY)
    base = alt.Chart(data)
    bars = base.mark_bar(cornerRadiusEnd=4, height={"band": .72}).encode(y=item, x=value, color=color, tooltip=tooltip)
    text = base.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(y=item, x=value, text=alt.Text("Value:Q", format=fmt))
    show_chart((bars + text).properties(height=30 * len(data) + (90 if groups else 60)))

def roc_chart(results):
    names = {name: f"{name} (AUC={r['roc_auc']:.3f})" for name, r in results.items()}
    data = pd.concat([pd.DataFrame({"Model": names[name], "FPR": r["roc"][0], "TPR": r["roc"][1]})
                      for name, r in results.items()], ignore_index=True)
    unit = alt.Scale(domain=[0, 1], nice=False)
    x = alt.X("FPR:Q", title="False positive rate", scale=unit)
    y = alt.Y("TPR:Q", title="True positive rate", scale=unit)
    palette = alt.Scale(domain=list(names.values()), range=MODEL_COLORS[:len(names)])
    color = alt.Color("Model:N", scale=palette)
    hover = alt.selection_point(name="roc_hover", on="pointerover", nearest=True, empty=False, clear="pointerout", fields=["Model", "FPR", "TPR"])
    chance = alt.Chart(pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})).mark_line(color=GREY, strokeDash=[4, 4], strokeWidth=1.5).encode(x=x, y=y)
    base = alt.Chart(data)
    lines = base.mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    targets = base.mark_point(opacity=0, size=300).encode(
        x=x, y=y, color=color,
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("FPR:Q", title="False positive rate", format=".3f"),
                 alt.Tooltip("TPR:Q", title="True positive rate", format=".3f")]).add_params(hover)
    marker = base.mark_circle(size=80, stroke="white", strokeWidth=2, opacity=1).encode(
        x=x, y=y, color=color).transform_filter(hover)
    show_chart((chance + lines + targets + marker).properties(height=440))

def metric_chart(results):
    labels = {"test_acc": "Test Acc", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
    data = pd.DataFrame([{"Model": name, "Metric": label, "Score": r[key]}
                         for name, r in results.items() for key, label in labels.items()])
    show_chart(alt.Chart(data).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("Metric:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelAngle=0, grid=False)),
        xOffset=alt.XOffset("Model:N", sort=list(results), scale=alt.Scale(paddingInner=.12)),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", sort=list(results), scale=alt.Scale(domain=list(results), range=MODEL_COLORS[:len(results)])),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Score:Q", format=".3f")],
    ).properties(height=340))

def confusion_chart(name, r, largest):
    """Two-by-two matrix; every matrix shares one colour scale so they compare directly."""
    cm, classes = r["cm"], ["Approve", "Repudiate"]
    data = pd.DataFrame([{"Actual": classes[a], "Predicted": classes[b], "Claims": int(cm[a, b]),
                          "Share": cm[a, b] / max(cm.sum(), 1)} for a in range(2) for b in range(2)])
    x = alt.X("Predicted:N", sort=classes, axis=alt.Axis(orient="bottom", labelAngle=0, grid=False))
    y = alt.Y("Actual:N", sort=classes, axis=alt.Axis(grid=False))
    cells = alt.Chart(data).mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("Claims:Q", legend=None, scale=alt.Scale(domain=[0, largest], range=["#EEF2FF", NAVY])),
        tooltip=[alt.Tooltip("Actual:N"), alt.Tooltip("Predicted:N"), alt.Tooltip("Claims:Q", format=","),
                 alt.Tooltip("Share:Q", title="Share of test set", format=".1%")])
    counts = alt.Chart(data).mark_text(fontSize=16, fontWeight="bold").encode(
        x=x, y=y, text=alt.Text("Claims:Q", format=","),
        color=alt.condition(f"datum.Claims > {largest * .55}", alt.value("white"), alt.value(INK)))
    show_chart((cells + counts).properties(height=220, title=alt.TitleParams(
        name, subtitle=f"acc={r['test_acc']:.2f} · recall={r['recall']:.2f}")))

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
st.html('<input type="checkbox" id="claims-navigation" checked aria-label="Show navigation" role="switch">\n<label for="claims-navigation" class="navigation-brand"><span class="nav-shield"></span><span class="nav-chevron"></span><span class="nav-brand-name">ClaimScope</span></label>')
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
    st.subheader(f"Repudiation rate by {sel}")
    rate_chart(g, overall)

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

    significant, not_significant = "Significant (p<0.05)", "Not significant (p≥0.05)"
    value_bars(assoc["Factor"], assoc["Cramér's V"], "Cramér's V (effect size)", fmt=".3f", groups={
        "assign": [not_significant if "—" in x else significant for x in assoc["significant (p<0.05)"]],
        "colors": {significant: NAVY, not_significant: GREY}})

    colA, colB = st.columns(2)

    # team / zone
    with colA:
        st.subheader("By TEAM / ZONE")
        minn = st.slider("Min claims per team", 10, 60, 25, key="zmin")
        g = df.groupby("ZONE")["REPUDIATED"].agg(["mean", "count"])
        g = g[g["count"] >= minn].sort_values("mean")
        rate_chart(g, overall)
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
        rate_chart(g, overall, horizontal=False)

        st.subheader("By INCOME QUARTILE")
        income_min = st.number_input("Minimum recorded annual income", min_value=0, value=0, step=10000, help="Explore income groups above this threshold. Model training still uses the complete dataset.")
        dz = df[df["PI_ANNUAL_INCOME"] > income_min].copy()
        st.caption(f"Only **{len(dz)}** of {len(df)} rows have income > 0 "
                   f"(rest are 0/missing) — read with caution.")
        if len(dz) > 20:
            dz["Q"] = pd.qcut(dz["PI_ANNUAL_INCOME"], 4, duplicates="drop")
            g = dz.groupby("Q", observed=True)["REPUDIATED"].agg(["mean", "count"])
            g.index = [interval_label(q) for q in g.index]
            rate_chart(g, overall, horizontal=False)

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
            value_bars(["Recorded", "Missing or zero"], [int((df[field]>0).sum()), int((df[field].fillna(0)<=0).sum())], "Claims")
        elif field in ["PI_AGE", "SUM_ASSURED"]:
            counts = df[field].value_counts(bins=12).sort_index()
            value_bars([interval_label(i) for i in counts.index], counts.to_numpy(), "Claims")
        else:
            limit = st.slider("Categories to preview", 4, 24, 12, step=4)
            counts = df[field].value_counts().head(limit)
            value_bars(counts.index, counts.to_numpy(), "Claims")
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
    roc_chart(results)
    st.caption("AUC well above 0.5 ⇒ there is learnable structure: the outcome is "
               "*predictable* from policy attributes. Combined with Tab 2, much of that "
               "structure is the processing channel (ZONE/STATE).")

    st.subheader("Metric comparison")
    metric_chart(results)

    st.subheader("Confusion matrices (test set)")
    cols = st.columns(2)
    largest = max(int(r["cm"].max()) for r in results.values())
    for i, (name, r) in enumerate(results.items()):
        with cols[i % 2]:
            confusion_chart(name, r, largest)

    # feature importance (RF)
    st.subheader("Feature importance (Random Forest)")
    rf = results["Random Forest"]["pipe"]
    ohe = rf.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
    X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
    feat = num + list(ohe.get_feature_names_out(cat))
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(15)
    value_bars(imp.index, imp.values, "Importance", fmt=".3f")
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
