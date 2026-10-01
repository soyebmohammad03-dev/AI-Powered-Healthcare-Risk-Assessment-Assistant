"""Shared UI layer: cached resources, design tokens, and small presentational components.

No model logic here: everything numeric comes from src/ or the precomputed artifacts.
"""
import json
from dataclasses import astuple
from html import escape
from pathlib import Path

import joblib
import plotly.graph_objects as go
import streamlit as st

from src import prediction, reliability as rel
from src.analysis import ARTIFACT
from src.data_loader import ROOT
from src.explainability import ModelExplainer
from src.prediction import ModelArtifactError, PatientInput, load_model
from src.preprocessing import CATEGORY_LABELS
from src.recommendations import DISCLAIMER, HIGHER_FROM, LOWER_BELOW

TRAIN_COMMAND = ("python -m src.train_models && python -m src.analysis && python -m src.reliability "
                 "&& python -m src.shift_analysis")
MODEL_NAMES = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}
UNITS = {"age": "years", "height": "cm", "weight": "kg", "ap_hi": "mmHg", "ap_lo": "mmHg"}
PAGES = {}  # filled by app.py so pages can link to each other

# ---- design tokens ----------------------------------------------------------------------------
TOKENS = {
    "light": dict(text="#111827", muted="#5B6472", faint="#8A93A0", border="#E2E6EB", surface="#FFFFFF",
                  sunken="#F1F3F6", primary="#0F6E6E", higher="#C2410C", lower="#0F766E",
                  low_band="#0F766E", mid_band="#B45309", high_band="#B91C1C", grid="#E9ECF0",
                  track=("#D7EFEC", "#F7E7C9", "#F6D4D4")),
    "dark": dict(text="#E5E7EB", muted="#A3ACB9", faint="#7C8594", border="#262E3A", surface="#121821",
                 sunken="#151B23", primary="#2DB4A6", higher="#FB923C", lower="#2DD4BF",
                 low_band="#2DD4BF", mid_band="#F59E0B", high_band="#F87171", grid="#232B36",
                 track=("#123B38", "#3D2F12", "#3F1D1D")),
}
MODEL_COLORS = {"logistic_regression": "#0F6E6E", "random_forest": "#6366F1", "xgboost": "#C2410C"}


def tokens() -> dict:
    return TOKENS["dark" if st.context.theme.type == "dark" else "light"]


def inject_css():
    t = tokens()
    st.html(f"""<style>
[data-testid="stMainBlockContainer"] {{ max-width: 1240px; padding-top: 4.4rem !important; padding-bottom: 4rem; }}
h1 {{ letter-spacing: -0.02em; }}
.eyebrow {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
           color: {t['primary']}; margin-bottom: .25rem; }}
.subtle {{ color: {t['muted']}; }}
.page-sub {{ color: {t['muted']}; font-size: 1.05rem; margin-top: -.4rem; margin-bottom: .6rem; }}
.section-label {{ font-size: .72rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
                 color: {t['muted']}; margin: .2rem 0 .5rem 0; }}
.chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600;
        border: 1px solid {t['border']}; color: {t['muted']}; margin-right: 6px; }}
.chip-band {{ font-size: .8rem; letter-spacing: .08em; padding: 3px 12px; }}
.hero-label {{ font-size: .72rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {t['muted']}; }}
.hero-number {{ font-size: 4.1rem; font-weight: 650; line-height: 1.05; letter-spacing: -0.03em;
               font-variant-numeric: tabular-nums; color: {t['text']}; }}
.meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .9rem 1.2rem;
             border-top: 1px solid {t['border']}; padding-top: .9rem; margin-top: .9rem; }}
.meta-k {{ font-size: .72rem; color: {t['faint']}; text-transform: uppercase; letter-spacing: .08em; }}
.meta-v {{ font-size: .93rem; font-weight: 550; color: {t['text']}; font-variant-numeric: tabular-nums; }}
.note {{ border-left: 3px solid {t['primary']}; padding: .55rem .9rem; background: {t['sunken']};
        border-radius: 0 .4rem .4rem 0; color: {t['text']}; font-size: .92rem; margin: .4rem 0 .8rem 0; }}
.kpi {{ border: 1px solid {t['border']}; border-radius: .5rem; padding: .8rem 1rem; background: {t['surface']}; }}
.kpi-k {{ font-size: .72rem; color: {t['faint']}; text-transform: uppercase; letter-spacing: .08em; }}
.kpi-v {{ font-size: 1.7rem; font-weight: 650; font-variant-numeric: tabular-nums; color: {t['text']}; }}
.kpi-d {{ font-size: .8rem; color: {t['muted']}; }}
.factor {{ display: grid; grid-template-columns: 1fr 90px 40px; align-items: center; gap: .6rem;
          padding: .32rem 0; border-bottom: 1px dashed {t['border']}; font-size: .9rem; }}
.factor:last-child {{ border-bottom: none; }}
.factor .bar {{ height: 7px; border-radius: 4px; }}
.factor .pct {{ text-align: right; color: {t['muted']}; font-variant-numeric: tabular-nums; font-size: .8rem; }}
.pipeline {{ display: flex; flex-wrap: wrap; gap: .45rem; align-items: stretch; margin: .3rem 0 1rem 0; }}
.step {{ flex: 1 1 calc(20% - .45rem); min-width: 150px; border: 1px solid {t['border']}; border-radius: .5rem; padding: .6rem .75rem;
        background: {t['surface']}; }}
.step-n {{ font-size: .68rem; color: {t['primary']}; font-weight: 700; letter-spacing: .08em; }}
.step-t {{ font-weight: 600; font-size: .92rem; color: {t['text']}; }}
.step-d {{ font-size: .78rem; color: {t['muted']}; line-height: 1.35; margin-top: .15rem; }}
.tagline {{ font-size: 1.15rem; font-weight: 600; color: {t['primary']}; margin: -.2rem 0 .6rem 0; }}
.foot b {{ color: {t['muted']}; }}
@media (max-width: 760px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 3.6rem !important; }}
  .hero-number {{ font-size: 3rem; }}
  .factor {{ grid-template-columns: 1fr 60px 36px; }}
}}
.foot {{ color: {t['faint']}; font-size: .8rem; border-top: 1px solid {t['border']}; padding-top: .8rem; margin-top: 2.5rem; }}
</style>""")


# ---- cached resources -------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained model…")
def load_engine(model_path: str):
    """Persisted pipeline + SHAP explainer, loaded once per server. Errors are not cached."""
    bundle = load_model(Path(model_path))
    return bundle, ModelExplainer(bundle)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def engine():
    """(bundle, explainer), or stop the page with a setup message instead of a traceback."""
    try:
        return load_engine(str(prediction.FINAL_MODEL_PATH))
    except ModelArtifactError:
        st.error("The trained model has not been generated yet, or the saved file cannot be used.",
                 icon=":material/error:")
    except Exception as exc:  # e.g. dataset download for the SHAP background failed while offline
        st.error(f"The assessment engine could not be started ({type(exc).__name__}).", icon=":material/error:")
    st.markdown("From the project folder, run this once, then reload the page:")
    st.code(TRAIN_COMMAND, language="bash")
    st.stop()


def analysis() -> dict:
    if not ARTIFACT.exists():
        st.error("The model analysis has not been generated yet.", icon=":material/error:")
        st.code(TRAIN_COMMAND, language="bash")
        st.stop()
    return load_json(str(ARTIFACT))


def metrics_report() -> dict:
    return load_json(str(ROOT / "artifacts" / "metrics.json"))


def artifact(name: str) -> dict | None:
    path = ROOT / "artifacts" / name
    return load_json(str(path)) if path.exists() else None


@st.cache_resource(show_spinner=False)
def candidates() -> dict:
    return rel.load_candidates()


@st.cache_resource(show_spinner=False)
def detector():
    return joblib.load(rel.DETECTOR_PATH) if rel.DETECTOR_PATH.exists() else None


def conformity(patient: PatientInput) -> dict | None:
    """Input-conformity check, or None if the detector has not been generated."""
    det = detector()
    return rel.input_conformity(patient, det) if det else None


@st.cache_data(show_spinner=False, max_entries=64)
def local_stability(values: tuple, _explainer, _pipeline) -> dict:
    return rel.local_stability(PatientInput(*values), _explainer, _pipeline)


def stability_for(patient: PatientInput, explainer) -> dict:
    return local_stability(astuple(patient), explainer, explainer.pipeline)


def step_note(bundle: dict) -> str:
    """Why the model estimate moves in steps, for the final model actually deployed."""
    if bundle["model_name"] == "logistic_regression":
        return "Flat steps come from the isotonic calibration." if bundle["calibration"] == "isotonic" else ""
    share = ((artifact("analysis.json") or {}).get("data_quality", {}).get("rounded_bp") or {}).get("ap_hi_120")
    detail = f" ({share:.0%} of systolic readings are exactly 120 mmHg)" if share is not None else ""
    return ("Steps and dips come from the tree model's split points; blood pressures in this data cluster at "
            f"round values{detail}, so splits sit near them.")


# ---- components -------------------------------------------------------------------------------
def page_header(eyebrow: str, title: str, subtitle: str):
    st.html(f"<div class='eyebrow'>{escape(eyebrow)}</div>")
    st.title(title, anchor=False)
    st.html(f"<div class='page-sub'>{escape(subtitle)}</div>")


def section(label: str):
    st.html(f"<div class='section-label'>{escape(label)}</div>")


def note(text: str):
    st.html(f"<div class='note'>{text}</div>")


def kpis(items: list[tuple[str, str, str]]):
    """Row of (label, value, detail) tiles."""
    cols = st.columns(len(items))
    for col, (k, v, d) in zip(cols, items):
        col.html(f"<div class='kpi'><div class='kpi-k'>{escape(k)}</div><div class='kpi-v'>{escape(v)}</div>"
                 f"<div class='kpi-d'>{escape(d)}</div></div>")


def band_name(p: float) -> str:
    return "Lower" if p < LOWER_BELOW else "Moderate" if p < HIGHER_FROM else "Higher"


def band_color(band: str) -> str:
    t = tokens()
    return {"Lower": t["low_band"], "Moderate": t["mid_band"], "Higher": t["high_band"]}[band]


def band_bar(p: float) -> str:
    t = tokens()
    lo, hi = LOWER_BELOW * 100, HIGHER_FROM * 100
    a, b, c = t["track"]
    return f"""
<div style="position:relative;margin:.9rem 0 .2rem 0">
  <div style="display:flex;height:10px;border-radius:5px;overflow:hidden">
    <div style="width:{lo}%;background:{a}"></div><div style="width:{hi - lo}%;background:{b}"></div><div style="flex:1;background:{c}"></div>
  </div>
  <div style="position:absolute;top:-6px;left:calc({p * 100:.1f}% - 2px);width:4px;height:22px;background:{t['text']};border-radius:2px"></div>
  <div style="display:flex;font-size:.72rem;color:{t['faint']};margin-top:6px;font-variant-numeric:tabular-nums">
    <span style="width:{lo}%">0% · Lower</span><span style="width:{hi - lo}%">{lo:.0f}% · Moderate</span><span>{hi:.0f}% · Higher</span>
  </div>
</div>"""


def factor_rows(contributions, color: str) -> str:
    rows = "".join(
        f"<div class='factor'><span>{escape(c.label)} <span class='subtle'>· {escape(c.display_value)}</span></span>"
        f"<div class='bar' style='width:{max(c.relative_importance, 0.03) * 100:.0f}%;background:{color}'></div>"
        f"<span class='pct'>{c.relative_importance:.0%}</span></div>" for c in contributions)
    return rows or "<div class='subtle' style='font-size:.9rem'>None</div>"


def readable(feature: str, value) -> str:
    if feature in CATEGORY_LABELS:
        return CATEGORY_LABELS[feature][int(value)]
    return f"{value:g} {UNITS[feature]}" if feature in UNITS else f"{value:g}"


def style_fig(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    t = tokens()
    fig.update_layout(**{
        "template": "plotly_dark" if st.context.theme.type == "dark" else "plotly_white",
        "height": height, "margin": dict(l=8, r=8, t=30, b=8), "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)", "font": dict(family="Inter, sans-serif", size=12.5, color=t["text"]),
        "legend": dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        "hoverlabel": dict(font_family="Inter, sans-serif"), **layout})  # caller overrides win
    fig.update_xaxes(gridcolor=t["grid"], zerolinecolor=t["border"])
    fig.update_yaxes(gridcolor=t["grid"], zerolinecolor=t["border"])
    return fig


def chart(fig: go.Figure, height: int = 360, **layout):
    st.plotly_chart(style_fig(fig, height, **layout), width="stretch", theme=None,
                    config={"displayModeBar": False})


def require_assessment():
    """Return the stored assessment or show an empty state linking back to Assess."""
    if "assessment" not in st.session_state:
        st.info("No assessment yet. Run one on the Assess page first; this page then explains it.",
                icon=":material/info:")
        if "assess" in PAGES:
            st.page_link(PAGES["assess"], label="Go to Assess", icon=":material/arrow_back:")
        st.stop()
    return st.session_state["assessment"]


def footer():
    st.html(f"<div class='foot'><b>AI-Powered Healthcare Risk Assessment Assistant</b> · educational prototype, "
            f"MIT licensed.<br>{escape(DISCLAIMER)} Model estimates come from one public research dataset "
            "and are not clinically validated.</div>")
