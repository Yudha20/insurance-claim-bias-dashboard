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
import base64
from pathlib import Path
import streamlit as st
import motion

# Show the skeleton before the slow imports below. Otherwise Streamlit draws its
# generic placeholder while they load. altair, scipy and scikit-learn are imported
# inside the functions that use them, so the first page does not wait for them.
st.set_page_config(page_title="ClaimScope · Insurance analytics", layout="wide",
                   initial_sidebar_state="expanded")
NAV_VIEWS = ["Claims", "Overview", "Disparities", "Data preparation", "Models", "Evaluation", "Findings"]
st.markdown("<style>" + Path(__file__).with_name("ui.css").read_text() + motion.motion_css() + "</style>", unsafe_allow_html=True)
st.html(motion.motion_script(), unsafe_allow_javascript=True)  # First, so it sets the theme before the skeleton paints.
st.html(motion.boot_html(NAV_VIEWS, NAV_VIEWS[0]))

import io
import html
import threading
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def product_font_css():
    faces = []
    for filename, weight, style in [("regular",400,"normal"),("bold",700,"normal"),("italic",400,"italic"),("bold-italic",700,"italic")]:
        data = base64.b64encode((Path(__file__).parent / "assets" / "fonts" / (filename + ".ttf")).read_bytes()).decode()
        faces.append(f"@font-face{{font-family:'Product Sans';font-style:{style};font-weight:{weight};font-display:swap;src:url(data:font/ttf;base64,{data}) format('truetype');}}")
    return "".join(faces)
st.markdown("<style>" + product_font_css() + "</style>", unsafe_allow_html=True)

DEFAULT_MODEL_CONFIG = dict(test_size=.25, seed=42, zone_top=15, state_top=15, occ_top=12, reason_top=12)

def mark_rendered():
    """Tell the browser this script run has finished drawing the page."""
    st.html(motion.render_marker(st.session_state.active_view, st.session_state.script_run))


# Colours below pass the colourblind-separation checks on a white surface.
NAVY, RED, GREY = "#375DFB", "#C25263", "#778196"
MODEL_COLORS = ["#375DFB", "#D9541F", "#128A5F", "#6B3FA0"]
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
                               symbolType="square", symbolOpacity=1, labelLimit=320)
             .configure_title(anchor="start", fontSize=14, fontWeight="normal", color=INK,
                              subtitleFontSize=12, subtitleColor=AXIS_TEXT))
    st.altair_chart(chart, width="stretch", theme=None)

def interval_label(interval):
    """Readable range for a pandas Interval, e.g. 45,000–90,000."""
    return f"{max(interval.left, 0):,.0f}–{interval.right:,.0f}"

def rate_chart(g, overall, horizontal=True):
    """Repudiation rate per group from a frame with mean/count, in display order, against the overall rate."""
    import altair as alt
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
    import altair as alt
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
    import altair as alt
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
    import altair as alt
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
    import altair as alt
    cm, classes = r["cm"], ["Approve", "Repudiate"]
    data = pd.DataFrame([{"Actual": classes[a], "Predicted": classes[b], "Claims": int(cm[a, b]),
                          "Share": cm[a, b] / max(cm.sum(), 1)} for a in range(2) for b in range(2)])
    x = alt.X("Predicted:N", sort=classes, axis=alt.Axis(orient="bottom", labelAngle=0, grid=False))
    y = alt.Y("Actual:N", sort=classes, axis=alt.Axis(grid=False))
    cells = alt.Chart(data).mark_rect(cornerRadius=4, stroke="white", strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("Claims:Q", legend=None, scale=alt.Scale(domain=[0, largest], range=["#EEF2FF", "#BFCDFF"])),
        tooltip=[alt.Tooltip("Actual:N"), alt.Tooltip("Predicted:N"), alt.Tooltip("Claims:Q", format=","),
                 alt.Tooltip("Share:Q", title="Share of test set", format=".1%")])
    # The scale stays light so one text colour reads on every cell, in both themes.
    counts = alt.Chart(data).mark_text(fontSize=16, fontWeight="bold", color=INK).encode(
        x=x, y=y, text=alt.Text("Claims:Q", format=","))
    show_chart((cells + counts).properties(height=220, title=alt.TitleParams(
        name, subtitle=f"acc={r['test_acc']:.2f} · recall={r['recall']:.2f}")))

def e(value):
    """Escape data values placed in explanation HTML; uploaded CSVs can contain anything."""
    return html.escape(str(value))

def page_intro(text):
    st.html(f'<div class="page-intro">{text}</div>')

def explain(how=None, data=None, caution=None):
    """One explanation style everywhere: how to read it, what this data shows, what to watch for."""
    rows = [("How to read", how), ("In this data", data), ("Keep in mind", caution)]
    st.html('<div class="explain">' + "".join(
        f'<div class="explain-label">{label}</div><div class="explain-text">{text}</div>'
        for label, text in rows if text) + "</div>")

def glossary(title, terms):
    with st.expander(title):
        st.html('<div class="glossary">' + "".join(
            f'<div class="glossary-term">{term}</div><div class="glossary-text">{text}</div>'
            for term, text in terms.items()) + "</div>")

def rate_insight(g, noun="groups"):
    """Plain-language summary of a mean/count frame; small groups are called out because they swing by chance."""
    if g.empty:
        return None
    if len(g) == 1:
        return f"Only <b>{e(g.index[0])}</b> is shown, at {g['mean'].iloc[0]*100:.1f}% of {g['count'].iloc[0]:,} claims."
    # Rank only groups large enough to mean something, unless that leaves nothing to compare.
    solid = g[g["count"] >= 30]
    ranked = solid if len(solid) >= 2 else g
    hi, lo = ranked["mean"].idxmax(), ranked["mean"].idxmin()
    hi_rate, lo_rate = g.loc[hi, "mean"], g.loc[lo, "mean"]
    lead = "Among groups with at least 30 claims, " if len(ranked) < len(g) else ""
    text = lead + (f"<b>{e(hi)}</b> is highest at {hi_rate*100:.1f}% ({g.loc[hi, 'count']:,} claims) and "
            f"<b>{e(lo)}</b> is lowest at {lo_rate*100:.1f}% ({g.loc[lo, 'count']:,} claims), "
            f"a gap of {(hi_rate-lo_rate)*100:.1f} percentage points")
    text += f", or {hi_rate/lo_rate:.1f}×." if lo_rate > 0 else "."
    small = g.index[g["count"] < 30].astype(str).tolist()
    if small:
        listed = ", ".join(e(x) for x in small[:3]) + (f" and {len(small)-3} more" if len(small) > 3 else "")
        text += f" {listed} {'has' if len(small) == 1 else 'have'} fewer than 30 claims, so treat {'that rate' if len(small) == 1 else 'those rates'} as rough."
    return text

FIELD_NOTES = {
    "PI_GENDER": "Gender of the insured person.",
    "EARLY_NON": "Whether the claim is classed as early, meaning it came soon after the policy began.",
    "MEDICAL_NONMED": "Whether the policy was issued with or without a medical examination.",
    "PAYMENT_MODE": "How premiums are paid: at regular intervals or as a single payment.",
    "ZONE": "The team, region or sales channel recorded against the policy.",
    "PI_STATE": "State recorded for the insured person.",
    "PI_OCCUPATION": "Occupation of the insured person.",
    "REASON_FOR_CLAIM": "Recorded reason for the claim. Blank values appear as Unknown.",
    "PI_AGE": "Age of the insured person.",
    "SUM_ASSURED": "The amount the policy covers.",
    "PI_ANNUAL_INCOME": "Annual income recorded for the insured person.",
}

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
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


# X and y are part of the cache key, so each dataset and set of category limits trains its own models.
@st.cache_resource(show_spinner=True, max_entries=8)
def train_models(X, y, num, cat, test_size, seed):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 precision_score, recall_score, roc_auc_score, roc_curve)
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=seed)
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
    from scipy.stats import chi2_contingency
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
st.session_state.script_run = st.session_state.get("script_run", 0) + 1
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
        st.caption("Replaces the sample data on every page for this session. Download the template for the 14 required columns. POLICY_STATUS must be Approved Death Claim or Repudiate Death.")
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
    mark_rendered()
    st.stop()
if "model_config" not in st.session_state:
    st.session_state.model_config = dict(DEFAULT_MODEL_CONFIG)

def close_model_settings():
    st.session_state.settings_open = False

@st.dialog("Model settings", width="medium", on_dismiss=close_model_settings)
def model_settings_dialog():
    cfg = st.session_state.model_config
    st.caption("Configure the training split and feature preparation. Changes apply only when saved.")
    with st.form("model_configuration", border=False):
        st.markdown("**Training**")
        split = st.slider("Holdout for evaluation", .15, .40, cfg["test_size"], .05, help="Fraction reserved for testing. Remaining claims train the models.")
        st.caption("A larger holdout gives a more reliable test but leaves fewer claims to learn from.")
        random_seed = st.number_input("Reproducibility seed", min_value=0, max_value=4294967295, value=cfg["seed"], step=1)
        st.caption("The same seed always gives the same split and results. Change it to check whether scores hold up on a different split.")
        st.markdown("**Category limits**")
        st.caption("Keep the most frequent categories and combine the rest as Other. Lower limits make simpler models; higher limits keep more detail but add rare categories.")
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
    st.html(motion.theme_toggle_html(), width="content")
if st.session_state.get("settings_open", False):
    model_settings_dialog()
overall = df["REPUDIATED"].mean()
for i, (col, label, value, tag, note) in enumerate(zip(st.columns(4),
    ["Total claims", "Approved claims", "Repudiated claims", "Income recorded"],
    [f"{len(df):,}", f"{(df.REPUDIATED==0).sum():,}", f"{df.REPUDIATED.sum():,}", f"{(df.PI_ANNUAL_INCOME.fillna(0)>0).mean()*100:.1f}%"],
    ["All records", f"{(1-overall)*100:.1f}%", f"{overall*100:.1f}%", "Data quality"],
    ["Current dataset", "Share of all claims", "Share of all claims", "Non-zero annual income"])):
    col.markdown(f'<div class="kpi" style="animation-delay:{i*40}ms"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}<span class="kpi-tag">{tag}</span></div></div>', unsafe_allow_html=True)
st.html(motion.page_skeletons_html(NAV_VIEWS))
with st.container(key="page_view"):
    if view == "Claims":
        page_intro("Every death claim in the current dataset, one row per claim. <b>Approved</b> means the claim was accepted. <b>Repudiated</b> means it was declined. The cards above always describe the whole dataset, on every page.")
        with st.container(border=True):
            st.subheader("Claim records", icon=":material/table_view:")
            explain(how="Search by policyholder or policy number, then narrow by outcome and state. Filters combine, and only affect this table and its export. <b>Sum assured</b> is the amount the policy covers.",
                    caution="The outcome is the decision on record. It does not show whether that decision was correct.")
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
                st.caption("What was cleaned when this dataset loaded. Every page uses the cleaned values.")
                for n in notes: st.write(n)

    # ============================== TAB 1 — DESCRIPTIVE =========================
    if view == "Overview":
        st.header("Descriptive Analysis — Cross-Tabulation vs POLICY_STATUS")
        page_intro("Start here to see how claim outcomes differ across any one attribute. Choose a field, compare the counts and rates, then look for groups far from the overall rate.")

        cat_options = [c for c in ["PI_GENDER", "EARLY_NON", "MEDICAL_NONMED", "PAYMENT_MODE",
                                   "ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"]
                       if c in df.columns]
        sel = st.selectbox("Cross-tabulate POLICY_STATUS against:", cat_options, index=0)

        explain(how=f"<b>{e(sel)}</b>: {FIELD_NOTES.get(sel, 'Selected field.')} <b>Counts</b> give the number of claims in each group. <b>Row %</b> gives each group's split between outcomes, so every row adds up to 100 and groups of different sizes can be compared fairly.")

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
        hidden = df[sel].nunique() - len(g)
        explain(how="Each bar is the share of that group's claims that were declined. The dashed line is the overall rate across all claims, so red bars sit above it and blue bars at or below it.",
                data=rate_insight(g),
                caution=(f"{hidden} {'group has' if hidden == 1 else 'groups have'} fewer than 5 claims and {'is' if hidden == 1 else 'are'} left out of this chart, but still counted in the tables above." if hidden else None))
        rate_chart(g, overall)

        st.subheader("Numeric summary by outcome")
        median_gap = None
        if df.REPUDIATED.nunique() == 2:
            approved_sum, repudiated_sum = [df.loc[df.REPUDIATED == k, "SUM_ASSURED"].median() for k in (0, 1)]
            median_gap = f"The median sum assured is {repudiated_sum:,.0f} on repudiated claims and {approved_sum:,.0f} on approved claims."
        explain(how="Compares age, sum assured and income between the two outcomes. <b>25%, 50% and 75%</b> are the values below which that share of claims falls, and 50% is the median. A mean far above its median points to a few very large values.",
                data=median_gap,
                caution="Zero income means not recorded, so income figures here are pulled down by missing values.")
        num_cols = [c for c in ["PI_AGE", "SUM_ASSURED", "PI_ANNUAL_INCOME"] if c in df.columns]
        st.dataframe(df.groupby(TARGET)[num_cols].describe().T.round(1), width='stretch')

    # ============================== TAB 2 — DIAGNOSTIC =========================
    if view == "Disparities":
        st.header("Diagnostic Analysis — Probing Differential Treatment")
        page_intro("This page ranks which attributes are most linked to repudiation, then looks closely at team, age and income. Use it to decide where a claim-file review is most worthwhile.")
        st.warning("**Interpretation guardrail:** a higher repudiation rate for a group is a "
                   "*disparity*, not automatically *bias*. The data does not record whether each "
                   "repudiation was justified (fraud / non-disclosure / policy condition). "
                   "Treat large gaps as **leads to audit**, confirm with claim-file review.")

        # association strength ranking
        st.subheader("Which factors are most associated with the outcome?")
        assoc_cols = [c for c in ["ZONE", "PI_STATE", "PAYMENT_MODE", "PI_OCCUPATION",
                                  "EARLY_NON", "MEDICAL_NONMED", "PI_GENDER"] if c in df.columns]
        rows = []
        for c in assoc_cols:
            chi2, p, dof, v, ncat = cramers_v(df, c)
            rows.append({"Factor": c, "Cramér's V": round(v, 3), "χ²": round(chi2, 1),
                         "dof": dof, "p-value": f"{p:.2e}", "categories": ncat,
                         "significant (p<0.05)": "Yes" if p < 0.05 else "No"})
        assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
        st.dataframe(assoc, width='stretch', hide_index=True)
        top_factor = assoc.iloc[0]
        weak = assoc.loc[assoc["significant (p<0.05)"] == "No", "Factor"].tolist()
        high_card = assoc[assoc["categories"] > 20]
        compact = assoc[(assoc["categories"] <= 10) & (assoc["significant (p<0.05)"] == "Yes")]
        top_v = top_factor["Cramér's V"]
        finding = f"<b>{e(top_factor['Factor'])}</b> has the strongest link, with V = {top_v:.3f}."
        if weak:
            finding += " No clear link for " + ", ".join(f"<b>{e(x)}</b>" for x in weak) + "."
        if not compact.empty:
            finding += f" Among fields with few categories, the strongest is <b>{e(compact.iloc[0]['Factor'])}</b>, a result less affected by small categories."
        explain(how="<b>Cramér's V</b> runs from 0, no link to the outcome, to 1, the outcome is fully determined by the field. Blue bars are statistically significant and grey bars are not.",
                data=finding,
                caution=("Fields with many small categories can score high partly by chance: " + ", ".join(f"{e(r.Factor)} ({r.categories} categories)" for r in high_card.itertuples()) + ". A strong link also says nothing about the cause." if not high_card.empty else "A strong link says nothing about the cause."))
        glossary("What the table columns mean", {
            "χ²": "How far the actual counts are from what you would expect if the field had no link to the outcome. Bigger means further.",
            "dof": "Degrees of freedom, set by the number of categories. The p-value calculation uses it.",
            "p-value": "How likely a link this strong would be by chance alone. Below 0.05 is treated as significant here.",
            "categories": "How many distinct values the field has.",
        })

        significant, not_significant = "Significant (p<0.05)", "Not significant (p≥0.05)"
        value_bars(assoc["Factor"], assoc["Cramér's V"], "Cramér's V (effect size)", fmt=".3f", groups={
            "assign": [significant if x == "Yes" else not_significant for x in assoc["significant (p<0.05)"]],
            "colors": {significant: NAVY, not_significant: GREY}})

        colA, colB = st.columns(2)

        # team / zone
        with colA:
            st.subheader("By TEAM / ZONE")
            minn = st.slider("Min claims per team", 10, 60, 25, key="zmin")
            g = df.groupby("ZONE")["REPUDIATED"].agg(["mean", "count"])
            g = g[g["count"] >= minn].sort_values("mean")
            explain(how=f"Showing {len(g)} of {df['ZONE'].nunique()} teams. Raise the minimum to hide small teams, where a few claims can move the rate sharply.",
                    data=rate_insight(g),
                    caution="Teams may handle different kinds of policy, so compare case mix before comparing teams.")
            rate_chart(g, overall)

        # age
        with colB:
            st.subheader("By AGE BAND")
            bins = [0, 18, 30, 40, 50, 60, 70, 200]
            labs = ["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
            ab = pd.cut(df["PI_AGE"], bins=bins, labels=labs, right=False)
            g = df.assign(AB=ab).groupby("AB", observed=True)["REPUDIATED"].agg(["mean", "count"])
            explain(how="Each column is an age group's repudiation rate, with its claim count beneath. The dashed line is the overall rate.",
                    data=rate_insight(g))
            rate_chart(g, overall, horizontal=False)

            st.subheader("By INCOME QUARTILE")
            income_min = st.number_input("Minimum recorded annual income", min_value=0, value=0, step=10000, help="Explore income groups above this threshold. Model training still uses the complete dataset.")
            dz = df[df["PI_ANNUAL_INCOME"] > income_min].copy()
            if len(dz) > 20:
                dz["Q"] = pd.qcut(dz["PI_ANNUAL_INCOME"], 4, duplicates="drop")
                g = dz.groupby("Q", observed=True)["REPUDIATED"].agg(["mean", "count"])
                g.index = [interval_label(q) for q in g.index]
                explain(how=f"Claims above the minimum are split into up to four bands of similar size. {len(dz):,} of {len(df):,} claims qualify. The dashed line stays at the rate for all claims.",
                        data=rate_insight(g),
                        caution=f"Income is not recorded for {(df.PI_ANNUAL_INCOME.fillna(0) <= 0).mean()*100:.0f}% of all claims, so these bands may not represent everyone.")
                rate_chart(g, overall, horizontal=False)
            else:
                st.info("Too few claims above this income to form bands. Lower the minimum.")

    # ============================== TAB 3 — FEATURE ENGINEERING ================
    if view == "Data preparation":
        st.header("Data preparation")
        page_intro("Check how trustworthy each field is and how the raw data becomes model input. <b>Field quality</b> looks at one field at a time, and <b>Model features</b> shows what the models actually receive.")
        detail, preview = st.tabs(["Field quality", "Model features"])
        with detail:
            field = st.selectbox("Inspect field", ["PI_ANNUAL_INCOME", "PI_AGE", "SUM_ASSURED", "ZONE", "PI_STATE", "PI_OCCUPATION", "REASON_FOR_CLAIM"])
            raw = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes is not None else pd.read_csv(Path(__file__).with_name("Insurance.csv"))
            changed = int((raw[field].astype(str).str.strip() != df[field].astype(str)).sum())
            explain(how=f"<b>{e(field)}</b>: {FIELD_NOTES.get(field, 'Selected field.')} <b>Missing values</b> are blank cells in the source file. <b>Distinct values</b> counts the different entries before cleaning.")
            m1,m2,m3 = st.columns(3)
            m1.metric("Records", f"{len(raw):,}")
            m2.metric("Missing values", f"{raw[field].isna().sum():,}")
            m3.metric("Distinct values", f"{raw[field].nunique():,}")
            if field == "PI_ANNUAL_INCOME":
                explain(caution="Zero income means not recorded, so the blank-cell count above can read 0 while many values are effectively missing. The models flag these records and fill in the median income.")
                value_bars(["Recorded", "Missing or zero"], [int((df[field]>0).sum()), int((df[field].fillna(0)<=0).sum())], "Claims")
            elif field in ["PI_AGE", "SUM_ASSURED"]:
                counts = df[field].value_counts(bins=12).sort_index()
                value_bars([interval_label(i) for i in counts.index], counts.to_numpy(), "Claims")
            else:
                limit = st.slider("Categories to preview", 4, 24, 12, step=4)
                explain(how="The most common values in the cleaned data, largest first. A long tail of small categories is what the limits in Model settings combine into Other.")
                counts = df[field].value_counts().head(limit)
                value_bars(counts.index, counts.to_numpy(), "Claims")
            explain(how="The table compares the first 20 rows before and after cleaning. Cleaning removes commas from amounts, trims spaces, fills blank text with Unknown and merges different capitalisations of ZONE.",
                    data=f"{changed:,} of {len(df):,} values differ from the source file after cleaning.")
            comparison = pd.DataFrame({"Original value":raw[field].astype(str).head(20), "Prepared value":df[field].astype(str).head(20)})
            st.dataframe(comparison, hide_index=True, width="stretch")
        with preview:
            explain(how="Each row is one claim as the models receive it, before numbers are scaled and categories encoded. Choose columns to preview, and use Model settings at the top right to change category limits.",
                    caution="Policy numbers and names are excluded, so no model can recognise an individual claim.")
            X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
            selected = st.multiselect("Preview columns", list(X.columns), default=list(X.columns)[:6])
            if selected:
                st.dataframe(X[selected].head(20), width="stretch", hide_index=True)
            else:
                st.info("Choose at least one column to preview model features.")
            pre = make_preprocessor(num, cat).fit(X)
            st.caption(f"{len(X):,} rows · {pre.transform(X).shape[1]} encoded features · identifiers excluded. Encoding gives each category its own yes-or-no column, which is why there are more features than fields.")
            with st.expander("Transformation details"):
                for note in notes: st.write(note)
                st.write("Numeric values are median-imputed and standardized. Categories are one-hot encoded; unseen categories are handled safely.")

    # ============================== TAB 4 — MODELS =============================
    if view in ["Models", "Evaluation"] and (len(df) < 24 or df.REPUDIATED.value_counts().min() < 6 or df.REPUDIATED.nunique() < 2):
        st.warning("Models need at least 24 records and 6 claims of each outcome. Use a larger dataset or return to Claims.")
        mark_rendered()
        st.stop()

    if view == "Models":
        st.header("Supervised Classification Models")
        page_intro("Four models learn to predict each claim's outcome from its attributes. This tests how predictable decisions are from the data. The models are not meant to decide claims.")
        glossary("The four models", {
            "KNN": "Looks at the 15 most similar past claims and takes their majority outcome.",
            "Decision Tree": "A flowchart of up to 6 yes-or-no questions. Easy to inspect, but less flexible.",
            "Random Forest": "300 trees, each up to 12 levels deep, voting together. Powerful, but can memorise its training data.",
            "Gradient Boosting": "A series of small trees, each correcting the last one's mistakes.",
        })
        glossary("What the metrics mean", {
            "Test Acc": "Share of unseen claims predicted correctly. Train Acc is the same measure on claims the model learned from.",
            "Precision": "Of the claims a model flags as repudiated, the share that really were.",
            "Recall": "Of the claims really repudiated, the share the model catches.",
            "F1": "A single balance of precision and recall.",
            "ROC-AUC": "How well a model ranks repudiated claims above approved ones. 0.5 is guessing and 1.0 is perfect.",
            "Overfit gap": "Train Acc minus Test Acc. A large gap means the model learned its training claims better than it can predict new ones.",
        })

        X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
        results, (ntr, nte, tr_pos, te_pos) = train_models(X, y, num, cat, test_size, int(seed))
        explain(how=f"Models learn from <b>{ntr:,}</b> claims ({tr_pos*100:.1f}% repudiated) and are scored on <b>{nte:,}</b> they have not seen ({te_pos*100:.1f}% repudiated). Both groups keep the same outcome mix. Change the split in Model settings.",
                caution="Decision Tree and Random Forest give the rarer outcome extra weight, which usually raises recall. KNN and Gradient Boosting do not, so compare their scores with that in mind.")

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
                     .highlight_max(axis=0, subset=["Overfit gap"], color="#f5b7b1").format("{:.3f}"),
                     width='stretch')

        best = metrics["ROC-AUC"].idxmax()
        baseline = max(te_pos, 1 - te_pos)
        best_acc = metrics["Test Acc"].idxmax()
        largest_gap, smallest_gap = metrics["Overfit gap"].idxmax(), metrics["Overfit gap"].idxmin()
        explain(how="Green marks the best score in each column. Red marks the largest overfit gap.",
                data=(f"<b>{e(best)}</b> ranks claims best, with ROC-AUC {metrics.loc[best, 'ROC-AUC']:.3f}. "
                      f"<b>{e(best_acc)}</b> has the top test accuracy at {metrics.loc[best_acc, 'Test Acc']*100:.1f}%, against {baseline*100:.1f}% from always predicting the more common outcome. "
                      f"<b>{e(largest_gap)}</b> has the largest overfit gap ({metrics.loc[largest_gap, 'Overfit gap']:.3f}) and <b>{e(smallest_gap)}</b> the smallest ({metrics.loc[smallest_gap, 'Overfit gap']:.3f})."),
                caution="Accuracy alone can mislead when one outcome is more common, so read it with recall and ROC-AUC.")

        st.subheader("Per-model classification report")
        pick = st.selectbox("Model", list(results.keys()), index=2)
        explain(how="Scores for each outcome separately, 0 to 1. <b>Support</b> is the number of test claims with that outcome. <b>Macro avg</b> treats both outcomes equally, and <b>weighted avg</b> accounts for how common each one is.",
                data=f"<b>{e(pick)}</b> catches {results[pick]['recall']*100:.0f}% of repudiated test claims, and {results[pick]['precision']*100:.0f}% of the claims it flags as repudiated really were.")
        st.code(results[pick]["report"])
        st.session_state["results"] = results   # share with eval tab

    # ============================== TAB 5 — EVALUATION =========================
    if view == "Evaluation":
        st.header("Model Evaluation — Stability & Errors")
        page_intro("The ROC, metric and confusion charts use test claims the models never saw during training, so they show how each model handles new claims. Feature importance shows what the Random Forest learned from its training claims.")
        X, y, num, cat = build_design_matrix(df, zone_top, state_top, occ_top, reason_top)
        results, _ = train_models(X, y, num, cat, test_size, int(seed))

        st.subheader("ROC curves")
        best_auc = max(results, key=lambda m: results[m]["roc_auc"])
        explain(how="Each curve shows the trade-off as a model flags more claims: catching more repudiations (up) at the cost of more false alarms on approved claims (right). Closer to the top-left is better. The dashed diagonal is random guessing.",
                data=f"<b>{e(best_auc)}</b> has the highest AUC at {results[best_auc]['roc_auc']:.3f}, the area under its curve. Hover over a curve for exact rates.",
                caution="A model scoring above 0.5 shows outcomes are partly predictable from these attributes. It does not show why, or whether the decisions were fair.")
        roc_chart(results)

        st.subheader("Metric comparison")
        top_recall = max(results, key=lambda m: results[m]["recall"])
        top_precision = max(results, key=lambda m: results[m]["precision"])
        explain(how="Taller is better. Definitions are on the Models page. Favour recall when missing a repudiation costs most, and precision when a false flag costs most.",
                data=f"<b>{e(top_recall)}</b> has the highest recall at {results[top_recall]['recall']:.2f}, and <b>{e(top_precision)}</b> the highest precision at {results[top_precision]['precision']:.2f}.")
        metric_chart(results)

        st.subheader("Confusion matrices (test set)")
        largest = max(int(r["cm"].max()) for r in results.values())
        fewest_missed = min(results, key=lambda m: results[m]["cm"][1, 0])
        missed_cm = results[fewest_missed]["cm"]
        explain(how="Rows are what actually happened and columns are what the model predicted. The top-left and bottom-right cells are correct. <b>Bottom-left</b> is a repudiation the model missed, and <b>top-right</b> is an approved claim it wrongly flagged. All four share one colour scale, and a stronger colour means more claims.",
                data=f"<b>{e(fewest_missed)}</b> misses the fewest repudiations, {missed_cm[1, 0]:,} of {missed_cm[1].sum():,}, while wrongly flagging {missed_cm[0, 1]:,} of {missed_cm[0].sum():,} approved claims.")
        cols = st.columns(2)
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
        by_field = imp.groupby([next((c for c in cat if name.startswith(c + "_")), name) for name in imp.index]).sum().sort_values(ascending=False)
        imp = imp.sort_values(ascending=False).head(15)
        explain(how="The 15 inputs the Random Forest relied on most. Each category is its own input, written as field_category, such as ZONE_AGENCY. Importance adds up to 1 across all inputs.",
                data="Added up by field, the largest shares belong to " + ", ".join(f"<b>{e(name)}</b> ({share*100:.0f}%)" for name, share in by_field.head(3).items()) + ".",
                caution="Importance shows what the model used, not what caused an outcome. Fields such as REASON_FOR_CLAIM and EARLY_NON may reflect information from the claim itself, which can make models look stronger than they would be earlier in the process.")
        value_bars(imp.index, imp.values, "Importance", fmt=".3f")

    # ============================== TAB 6 — FINDINGS ===========================
    if view == "Findings":
        st.header("Findings & Recommendations")
        page_intro("The main takeaways from the current dataset and what to check next. The figures recalculate if you upload a different file.")
        st.write(f"Of {len(df):,} claims, {int(df.REPUDIATED.sum()):,} were repudiated ({overall:.1%}).")
        group = df.groupby("ZONE").REPUDIATED.agg(["mean", "count"])
        eligible = group[group["count"] >= 20].sort_values("mean")
        if len(eligible) >= 2:
            st.write(f"Among teams with at least 20 claims, repudiation ranges from {eligible['mean'].min():.1%} to {eligible['mean'].max():.1%}. Compare case mix and documented grounds before interpreting this gap.")
        else:
            st.info("Too few teams have 20 claims for a stable team comparison.")
        st.write(f"Annual income is missing or zero for {(df.PI_ANNUAL_INCOME.fillna(0) <= 0).mean():.1%} of records. Treat income comparisons cautiously.")
        st.subheader("Audit next steps")
        st.markdown("""1. **Review a stratified sample from teams with high and low repudiation rates.** Sampling from both ends shows whether similar claims are being decided differently.
2. **Verify documented policy grounds and case mix with an independent reviewer.** The dataset has no record of why each claim was declined, and case mix means the kinds of policy and claim a team handles.
3. **Improve missing income capture before drawing income-related conclusions.** Comparisons only cover claims with income recorded.
4. **Compare models with and without geography and processing-channel attributes.** If accuracy drops sharply without them, outcomes depend heavily on where a claim was handled.""")
        explain(caution="Disparities are leads for audit, not proof of bias. This dataset does not establish whether each repudiation was justified, so a high rate may reflect different risks or inconsistent decisions.")
        st.caption("The cleaned dataset includes the merged team names and two added columns: INCOME_MISSING flags unrecorded income, and REPUDIATED marks declined claims with 1.")
        st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode(), "insurance_cleaned.csv", "text/csv",
                           icon=":material/download:", on_click=download_feedback)

    st.caption("Built with Streamlit + scikit-learn · disparities ≠ proof of bias · "
               "validate against documented repudiation grounds before acting.")

mark_rendered()

def warm_up_models():
    """Train the default models early, so the first Models visit does not wait for them."""
    c = DEFAULT_MODEL_CONFIG
    try:
        sample, _ = load_and_engineer(None)
        X, y, num, cat = build_design_matrix(sample, c["zone_top"], c["state_top"], c["occ_top"], c["reason_top"])
        train_models(X, y, num, cat, c["test_size"], c["seed"])
    except Exception:
        pass  # The Models page trains on demand if this fails.

@st.cache_resource(show_spinner=False)
def start_model_warm_up():
    """Runs once per server process, after the first page has already been sent."""
    threading.Thread(target=warm_up_models, daemon=True).start()

start_model_warm_up()
