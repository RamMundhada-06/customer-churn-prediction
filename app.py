"""
app.py – ChurnGuard AI  (Streamlit)
Light, calm UI — indigo/slate palette, white cards, soft shadows.
4 models: Logistic Regression | Decision Tree | Random Forest | KNN
"""

import json, pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS – light, calm, professional
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #f5f7ff;
    color: #1e293b;
}
[data-testid="stSidebar"] {
    background: #eef0fb;
    border-right: 1px solid #dde1f5;
}

/* ── White cards with soft shadow ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e6f8;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px rgba(79,70,229,0.07);
}
.card h3 {
    margin-top: 0;
    color: #4f46e5;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(100deg, #6366f1 0%, #818cf8 60%, #a5b4fc 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 20px rgba(99,102,241,0.25);
}
.hero h1 { margin: 0; font-size: 1.8rem; color: #fff; font-weight: 800; }
.hero p  { margin: .3rem 0 0; color: rgba(255,255,255,0.88); font-size: .95rem; }

/* ── Risk badges ── */
.risk-high   { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5;
               border-radius:8px; padding:.3rem .75rem; font-weight:700; }
.risk-medium { background:#fef3c7; color:#92400e; border:1px solid #fcd34d;
               border-radius:8px; padding:.3rem .75rem; font-weight:700; }
.risk-low    { background:#d1fae5; color:#065f46; border:1px solid #6ee7b7;
               border-radius:8px; padding:.3rem .75rem; font-weight:700; }

/* ── Large probability number ── */
.prob-label { font-size: 2.8rem; font-weight: 800; text-align: center; }

/* ── Inputs – keep light style ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #f8f9ff !important;
    border-color: #c7cdf0 !important;
    border-radius: 9px !important;
    color: #1e293b !important;
}

/* ── Slider ── */
div[data-baseweb="slider"] > div > div { background: #6366f1 !important; }

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    color: #fff; border: none; border-radius: 10px;
    padding: .7rem 2rem; font-size: 1rem; font-weight: 700;
    width: 100%; transition: all .2s;
    box-shadow: 0 3px 12px rgba(79,70,229,0.3);
}
div.stButton > button:hover {
    box-shadow: 0 5px 18px rgba(79,70,229,0.45);
    transform: translateY(-1px);
}

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: #e8eaf8; border-radius: 10px; padding: 4px;
}
button[data-baseweb="tab"] {
    border-radius: 8px !important; color: #4f46e5 !important; font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: #fff !important; color: #3730a3 !important;
    box-shadow: 0 1px 6px rgba(79,70,229,0.15) !important;
}

/* ── Sidebar radio ── */
div[data-testid="stRadio"] label { color: #3730a3 !important; font-weight: 600; }

/* ── Metric values ── */
[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #3730a3 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Divider ── */
hr { border-color: #e2e6f8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "Logistic Regression": {"key": "lr",  "icon": "📈", "color": "#3b82f6"},
    "Decision Tree":       {"key": "dt",  "icon": "🌿", "color": "#22c55e"},
    "Random Forest":       {"key": "rf",  "icon": "🌲", "color": "#16a34a"},
    "KNN":                 {"key": "knn", "icon": "🔍", "color": "#a855f7"},
}

META_PATH = pathlib.Path("model/model_meta.json")

# ─────────────────────────────────────────────────────────────────────────────
# Load metadata
# ─────────────────────────────────────────────────────────────────────────────
meta = {}
if META_PATH.exists():
    meta = json.loads(META_PATH.read_text())

all_results = meta.get("all_results", {})
thresh_min  = meta.get("thresh_min", 0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Cache each model separately
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(key: str):
    p = pathlib.Path(f"model/{key}_model.pkl")
    if p.exists():
        return joblib.load(p)
    raise FileNotFoundError(
        f"Model file not found: {p} — run `python train_model.py` first."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard AI")
    st.caption("Customer churn prediction · IBM Telco dataset")
    st.divider()

    # ── Model selector ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Choose a Model")
    available = [m for m in MODEL_DISPLAY
                 if pathlib.Path(f"model/{MODEL_DISPLAY[m]['key']}_model.pkl").exists()]
    if not available:
        available = list(MODEL_DISPLAY.keys())

    selected_model = st.radio(
        "Model",
        options=available,
        format_func=lambda m: f"{MODEL_DISPLAY[m]['icon']}  {m}",
        index=min(2, len(available) - 1),
        label_visibility="collapsed",
    )

    m_info    = MODEL_DISPLAY[selected_model]
    m_results = all_results.get(selected_model, {})
    default_thresh = float(m_results.get("Threshold", thresh_min))

    st.divider()

    # ── Threshold slider ─────────────────────────────────────────────────────
    st.markdown("### 🎚️ Decision Threshold")
    threshold = st.slider(
        "Threshold", min_value=0.30, max_value=0.80,
        value=default_thresh, step=0.01,
        label_visibility="collapsed",
        help=(f"Probability ≥ this → predicted churn. "
              f"Tuned default for {selected_model}: {default_thresh}"),
    )
    st.caption(
        f"Tuned default: **{default_thresh}** &nbsp;|&nbsp; Active: **{threshold}**"
    )

    st.divider()

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.markdown(f"### {m_info['icon']} {selected_model} — Test Metrics")
    if m_results:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC",   m_results.get("ROC-AUC",  "—"))
        c2.metric("F1 Score",  m_results.get("F1",       "—"))
        c1.metric("Precision", m_results.get("Precision","—"))
        c2.metric("Recall",    m_results.get("Recall",   "—"))
        c1.metric("Accuracy",  m_results.get("Accuracy", "—"))
    else:
        st.info("Run `python train_model.py` to see metrics.")

    st.divider()
    st.markdown("### ℹ️ How to use")
    st.write(
        "1. Pick a **model** above  \n"
        "2. Adjust the **threshold** slider  \n"
        "3. Fill in customer details  \n"
        "4. Click **Predict Churn**"
    )
    st.caption("sklearn + Streamlit · IBM Telco dataset")

# ─────────────────────────────────────────────────────────────────────────────
# Load selected model
# ─────────────────────────────────────────────────────────────────────────────
try:
    model = load_model(m_info["key"])
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🛡️ ChurnGuard AI</h1>
  <p>Using &nbsp;<b>{m_info['icon']} {selected_model}</b>
  &nbsp;·&nbsp; Threshold: <b>{threshold}</b>
  &nbsp;·&nbsp; Fill in customer details and click Predict.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_compare, tab_about = st.tabs([
    "🔮  Predict Churn", "📊  Model Comparison", "📖  About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:

    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="card"><h3>👤 Customer Demographics</h3>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    gender     = d1.selectbox("Gender",         ["Male", "Female"])
    senior     = d2.selectbox("Senior Citizen",  ["No", "Yes"])
    partner    = d3.selectbox("Partner",         ["Yes", "No"])
    dependents = d4.selectbox("Dependents",      ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12,
                       help="How long this customer has been with the company")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Services ──────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><h3>📡 Services Subscribed</h3>', unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    phone    = s1.selectbox("Phone Service",    ["Yes", "No"])
    multi    = s2.selectbox("Multiple Lines",   ["No",  "Yes"])
    internet = s3.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    security = s4.selectbox("Online Security",  ["No",  "Yes"])
    backup   = s5.selectbox("Online Backup",    ["No",  "Yes"])

    s6, s7, s8, s9 = st.columns(4)
    protection = s6.selectbox("Device Protection", ["No", "Yes"])
    tech       = s7.selectbox("Tech Support",       ["No", "Yes"])
    tv         = s8.selectbox("Streaming TV",       ["No", "Yes"])
    movies     = s9.selectbox("Streaming Movies",   ["No", "Yes"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Billing ───────────────────────────────────────────────────────────────
    st.markdown('<div class="card"><h3>💳 Billing & Contract</h3>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    contract  = b1.selectbox("Contract Type",     ["Month-to-month", "One year", "Two year"])
    paperless = b2.selectbox("Paperless Billing",  ["Yes", "No"])
    payment   = b3.selectbox("Payment Method",    [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"])
    b4, b5 = st.columns(2)
    monthly = b4.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
    total   = b5.number_input("Total Charges ($)",   0.0, 10000.0,
                               float(round(tenure * 70, 2)), step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────────────────────
    predict_clicked = st.button(
        f"{m_info['icon']}  Predict with {selected_model}",
        use_container_width=True,
    )

    if predict_clicked:
        row = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }])

        prob = float(model.predict_proba(row)[0, 1])
        pct  = round(prob * 100, 1)

        # Risk level
        if prob >= threshold + 0.15:
            risk_label, risk_cls, risk_icon = "HIGH RISK",   "risk-high",   "🔴"
            tip = "**Action needed:** Offer a loyalty discount, service upgrade, or a dedicated account manager."
        elif prob >= threshold:
            risk_label, risk_cls, risk_icon = "MEDIUM RISK", "risk-medium", "🟠"
            tip = "**Watch closely:** Schedule a proactive check-in call and highlight recent service improvements."
        else:
            risk_label, risk_cls, risk_icon = "LOW RISK",    "risk-low",    "🟢"
            tip = "**Keep it up:** Maintain service quality. Consider a referral reward to deepen loyalty."

        # ── Result card ───────────────────────────────────────────────────────
        st.divider()
        bar_color = {
            "risk-high":   "#ef4444",
            "risk-medium": "#f59e0b",
            "risk-low":    "#10b981",
        }[risk_cls]

        st.markdown(f"""
<div class="card" style="border-left: 4px solid {bar_color};">
  <h3>📊 Prediction Result
    <span style="font-size:.8rem;font-weight:400;text-transform:none;color:#64748b;">
      via {m_info['icon']} {selected_model} &nbsp;|&nbsp; threshold = {threshold}
    </span>
  </h3>
</div>""", unsafe_allow_html=True)

        r1, r2, r3 = st.columns([1, 2, 1])

        with r1:
            st.markdown(
                f"<div style='text-align:center;padding-top:1.2rem;'>"
                f"<div style='font-size:3.5rem;'>{risk_icon}</div>"
                f"<span class='{risk_cls}'>{risk_label}</span>"
                f"</div>", unsafe_allow_html=True)

        with r2:
            st.markdown(
                f"<div class='prob-label' style='color:{bar_color};'>{pct}%</div>",
                unsafe_allow_html=True)
            st.caption("Churn Probability")
            st.progress(prob)
            st.markdown(
                f"<small style='color:#64748b;'>Threshold: <b>{threshold}</b>"
                f" &nbsp;|&nbsp; Model: <b>{selected_model}</b></small>",
                unsafe_allow_html=True)

        with r3:
            st.metric("Tenure",   f"{tenure} mo")
            st.metric("Monthly",  f"${monthly:.0f}")
            st.metric("Contract", contract.split()[0])

        # ── Retention tip ─────────────────────────────────────────────────────
        tip_bg = {
            "risk-high":   "rgba(239,68,68,0.08)",
            "risk-medium": "rgba(245,158,11,0.08)",
            "risk-low":    "rgba(16,185,129,0.08)",
        }[risk_cls]
        st.markdown(f"""
<div style="background:{tip_bg};border-left:4px solid {bar_color};
            border-radius:8px;padding:1rem 1.2rem;margin-top:.8rem;">
  <strong>💡 Retention Tip</strong><br>{tip}
</div>""", unsafe_allow_html=True)

        # ── Risk factor breakdown ─────────────────────────────────────────────
        st.markdown("#### 🔍 Key Risk Factors")
        factors = []
        if contract == "Month-to-month":
            factors.append(("📄 Month-to-month contract",
                            "No commitment lock-in — strongest individual churn predictor."))
        if internet == "Fiber optic" and security == "No":
            factors.append(("🔒 No Online Security (Fiber)",
                            "Fiber customers without security churn ~2× more."))
        if tenure < 12:
            factors.append(("⏱️ Short tenure (<12 months)",
                            "New customers are the highest churn risk — still evaluating the service."))
        if monthly > 70 and contract == "Month-to-month":
            factors.append(("💸 High charge + no commitment",
                            "Price-sensitive customers on flexible plans often switch for a better deal."))
        if tech == "No" and internet != "No":
            factors.append(("🛠️ No Tech Support",
                            "Lack of support increases frustration-driven churn, especially on Fiber."))
        if payment == "Electronic check":
            factors.append(("💳 Electronic check payment",
                            "Historically the payment method with the highest churn rate in this dataset."))
        if senior == "Yes":
            factors.append(("👴 Senior Citizen",
                            "Senior customers show a slightly elevated average churn rate."))

        if factors:
            for title, desc in factors:
                st.markdown(f"""
<div style="background:#fef2f2;border:1px solid #fecaca;
            border-radius:10px;padding:.7rem 1rem;margin:.35rem 0;">
  <b style="color:#991b1b;">{title}</b><br>
  <small style="color:#475569;">{desc}</small>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;
            border-radius:10px;padding:.7rem 1rem;">
  <b style="color:#166534;">No major individual risk factors detected.</b>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="card"><h3>📊 All Models — Side by Side</h3>', unsafe_allow_html=True)
    st.write("GridSearchCV scored by **F1** (5-fold stratified CV). "
             "Each threshold independently tuned per model (min 0.45).")
    st.markdown("</div>", unsafe_allow_html=True)

    if all_results:
        rows = []
        for name, r in all_results.items():
            icon = MODEL_DISPLAY.get(name, {}).get("icon", "")
            rows.append({
                "Model":     f"{icon} {name}",
                "F1":        r.get("F1",       "—"),
                "Precision": r.get("Precision","—"),
                "Recall":    r.get("Recall",   "—"),
                "Accuracy":  r.get("Accuracy", "—"),
                "ROC-AUC":   r.get("ROC-AUC",  "—"),
                "Threshold": r.get("Threshold","—"),
            })
        df_cmp = pd.DataFrame(rows).set_index("Model")

        def highlight_best(s):
            try:
                is_max = s.astype(float) == s.astype(float).max()
                return ["background-color:#eef2ff;font-weight:700;color:#3730a3"
                        if v else "" for v in is_max]
            except Exception:
                return [""] * len(s)

        styled = (df_cmp.style
                  .apply(highlight_best,
                         subset=["F1","Precision","Recall","Accuracy","ROC-AUC"])
                  .format("{:.4f}",
                          subset=["F1","Precision","Recall","Accuracy","ROC-AUC"]))
        st.dataframe(styled, use_container_width=True)

        # ── Mini model cards ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Model Cards")
        cols = st.columns(len(all_results))
        for col, (name, r) in zip(cols, all_results.items()):
            icon   = MODEL_DISPLAY.get(name, {}).get("icon", "")
            color  = MODEL_DISPLAY.get(name, {}).get("color", "#4f46e5")
            active = name == selected_model
            border = f"2px solid {color}" if active else "1px solid #e2e6f8"
            bg     = f"rgba(79,70,229,0.04)" if active else "#fff"
            col.markdown(f"""
<div style="background:{bg};border:{border};border-radius:14px;
            padding:1rem;text-align:center;box-shadow:0 2px 8px rgba(79,70,229,0.08);">
  <div style="font-size:2rem;">{icon}</div>
  <div style="font-weight:700;color:{color};margin:.25rem 0;font-size:.95rem;">{name}</div>
  <div style="font-size:.7rem;color:#94a3b8;margin-top:.4rem;">ROC-AUC</div>
  <div style="font-size:1.3rem;font-weight:800;color:#1e293b;">{r.get('ROC-AUC','—')}</div>
  <div style="font-size:.7rem;color:#94a3b8;margin-top:.3rem;">F1</div>
  <div style="font-size:1.1rem;font-weight:700;color:{color};">{r.get('F1','—')}</div>
  <div style="font-size:.7rem;color:#94a3b8;margin-top:.3rem;">Threshold</div>
  <div style="font-size:1rem;font-weight:600;color:#f59e0b;">{r.get('Threshold','—')}</div>
  {"<div style='margin-top:.5rem;font-size:.7rem;background:#eef2ff;color:#3730a3;border-radius:6px;padding:.2rem .6rem;font-weight:700;'>ACTIVE</div>" if active else ""}
</div>""", unsafe_allow_html=True)
    else:
        st.info("No results found. Run `python train_model.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
<div class="card">
<h3>🤖 About ChurnGuard AI</h3>

**ChurnGuard AI** predicts whether a telecom customer is likely to cancel their subscription,
using an ML pipeline trained on the IBM Telco Customer Churn dataset (~7,000 records, 26.5% churn rate).

#### ML Pipeline
1. **Cleaning** — TotalCharges → numeric; "No internet/phone service" → "No"; Churn → 0/1
2. **Preprocessing** — `StandardScaler` for numerics; `OneHotEncoder` for categoricals (no data leakage)
3. **Training** — 4 classifiers tuned via `GridSearchCV` scored by **F1** (5-fold stratified CV)
4. **Threshold tuning** — each model's decision cutoff is independently tuned (≥ 0.45) to maximise F1

#### Models

| Model | Strengths | Weaknesses |
|---|---|---|
| 📈 Logistic Regression | Fast, interpretable, good baseline | Assumes linear decision boundary |
| 🌿 Decision Tree | Interpretable rules, handles non-linearity | Prone to overfitting without pruning |
| 🌲 Random Forest | Robust, low variance, strong on tabular data | Slower, less interpretable |
| 🔍 KNN | Non-parametric, intuitive | Slow at inference, sensitive to irrelevant features |

</div>
<div class="card" style="margin-top:1rem;">
<h3>📚 Features Used (19 total)</h3>

| Category | Features |
|---|---|
| Demographics | Gender, Senior Citizen, Partner, Dependents, Tenure |
| Services | Phone, Multiple Lines, Internet, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| Billing | Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges |

</div>
<div class="card">
<h3>⚠️ Disclaimer</h3>
For demonstration and educational purposes only.
Predictions are probabilistic — combine with domain expertise for real business decisions.
</div>
""", unsafe_allow_html=True)
