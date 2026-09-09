"""
Streamlit frontend for the Stroke Risk Prediction demo.
Talks only to the FastAPI backend (no ML logic lives here).

Run:
    streamlit run app.py
Make sure the backend is running first:
    uvicorn main:app --port 8000   (from the backend/ folder)
"""
import os
from datetime import date

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

#FastAPI endpoint
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="StrokeGuard | Stroke Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Light medical theme - Custom page

st.markdown("""
<style>
:root {
    --sg-primary: #0f766e;
    --sg-primary-light: #ccfbf1;
    --sg-accent: #2563eb;
    --sg-danger: #dc2626;
    --sg-warning: #d97706;
    --sg-bg: #f4faf9;
    --sg-card: #ffffff;
}

/* =========================
   APP BACKGROUND
   ========================= */
.stApp {
    background: linear-gradient(180deg, #f0fbf9 0%, #f7fafc 100%);
}

/* =========================
   HERO
   ========================= */
.sg-hero {
    background: linear-gradient(
        120deg,
        #0f766e 0%,
        #14b8a6 60%,
        #5eead4 100%
    );
    border-radius: 18px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(15, 118, 110, 0.18);
}

.sg-hero h1 {
    margin: 0 0 6px 0;
    font-size: 2.0rem;
}

.sg-hero p {
    margin: 0;
    opacity: 0.95;
    font-size: 1.02rem;
}

/* =========================
   CARD
   ========================= */
.sg-card {
    background: var(--sg-card);
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}

.sg-metric-card {
    background: var(--sg-card);
    border-left: 5px solid var(--sg-primary);
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}

/* =========================
   GENERAL TEXT
   ========================= */
.stMarkdown p {
    font-size: 17px;
    line-height: 1.6;
}

/* =========================
   TAB STYLE
   ========================= */

/* Tab button */
button[data-baseweb="tab"] {
    padding: 12px 24px !important;
    color: #334155 !important;
}

/* Actual tab text */
button[data-baseweb="tab"] p {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] p {
    color: #0f766e !important;
    font-weight: 700 !important;
}

/* Tab container */
div[data-baseweb="tab-list"] {
    gap: 8px !important;
}

/* Active underline */
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 3px solid #0f766e !important;
}

/* =========================
   INPUT LABELS
   ========================= */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stRadio label,
.stDateInput label {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* =========================
   BADGES
   ========================= */
.sg-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 16px !important;
}

.sg-badge-low {
    background: #dcfce7;
    color: #15803d;
}

.sg-badge-moderate {
    background: #fef3c7;
    color: #b45309;
}

.sg-badge-high {
    background: #fee2e2;
    color: #b91c1c;
}

/* =========================
   PREDICTION RESULT
   ========================= */
.sg-result-yes,
.sg-result-no {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 16px !important;
    font-weight: 600;
}

.sg-result-yes {
    background: #fee2e2;
    color: #b91c1c;
}

.sg-result-no {
    background: #dcfce7;
    color: #15803d;
}

/* =========================
   RESULT INFORMATION
   ========================= */
.sg-result-content {
    font-size: 17px;
    line-height: 1.8;
}

.sg-result-content .info-label {
    font-size: 17px;
    font-weight: 600;
}

.sg-result-content .info-value {
    font-size: 18px;
}

/* =========================
   STREAMLIT METRIC
   ========================= */
div[data-testid="stMetricValue"] {
    color: #0f766e;
    font-size: 36px !important;
}

div[data-testid="stMetricLabel"] {
    font-size: 17px !important;
    font-weight: 700 !important;
}

/* =========================
   PLACEHOLDER
   ========================= */
.sg-placeholder {
    border: 2px dashed #cbd5e1;
    border-radius: 14px;
    padding: 40px 24px;
    text-align: center;
    color: #64748b;
    font-size: 17px;
    line-height: 1.6;
}

/* =========================
   DISCLAIMER
   ========================= */
.sg-disclaimer {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 15px !important;
    line-height: 1.6 !important;
    color: #9a3412;
}

/* =========================
   BUTTON
   ========================= */
.stButton > button,
.stFormSubmitButton > button {
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* =========================
   SELECTBOX / INPUT TEXT
   ========================= */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    font-size: 16px !important;
}

/* =========================
   RADIO
   ========================= */
.stRadio label {
    font-size: 16px !important;
}

/* =========================
   PROGRESS BAR
   ========================= */
div[data-testid="stProgress"] {
    margin-top: 8px;
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)


# Backend helpers (cached so the app stays snappy)

@st.cache_data(ttl = 300)
def get_options():
    r = requests.get(f"{BACKEND_URL}/meta/options", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl = 300)
def get_models():
    r = requests.get(f"{BACKEND_URL}/meta/models", timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl = 300)
def get_statistics():
    r = requests.get(f"{BACKEND_URL}/meta/statistics", timeout=10)
    r.raise_for_status()
    return r.json()


def call_predict(payload: dict):
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def backend_is_up() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# TITLE

st.markdown("""
<div class="sg-hero">
    <h1>🩺 StrokeGuard — Stroke Risk Prediction</h1>
    <p>An educational demo that explores the stroke dataset and estimates stroke risk using machine learning.</p>
</div>
""", unsafe_allow_html=True)

if not backend_is_up():
    st.error(
        f"Can't reach the prediction API at **{BACKEND_URL}**. "
        "Start it first with `uvicorn main:app --port 8000` from the `backend/` folder, "
        "then refresh this page."
    )
    st.stop()

tab1, tab2 = st.tabs(["📊 Stroke Rate Analytics", "🧪 Stroke Risk Prediction"])


# TAB 1 — Stroke Rate Analytics
with tab1: 
    stats = get_statistics()
    st.subheader("📊 Stroke Rate Analytics")
    
    # Overview
    c1, c2 = st.columns(2)
    c1.metric( "Total Patients", f"{stats['n_rows']:,}")
    c2.metric("Overall Stroke Rate", f"{stats['overall_stroke_rate'] * 100:.2f}%")

    st.divider()

    # AGE GROUP
    st.markdown("### Age Group")
    age_df = pd.DataFrame(stats["age_group_vs_stroke"])
    age_df["Stroke Rate (%)"] = age_df["stroke_rate"] * 100
    st.bar_chart(
        age_df,
        x="Age Group",
        y="Stroke Rate (%)",
        horizontal=True
    )

    # HYPERTENSION + HEART DISEASE
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Hypertension")
        df = pd.DataFrame(stats["hypertension_vs_stroke"])
        df["Hypertension"] = df["Hypertension"].map({
            0: "No",
            1: "Yes"
        })
        df["Stroke Rate (%)"] = df["stroke_rate"] * 100
        st.bar_chart(
            df,
            x="Hypertension",
            y="Stroke Rate (%)"
        )

    with col2:
        st.markdown("### Heart Disease")
        df = pd.DataFrame(stats["heart_disease_vs_stroke"])
        df["Heart Disease"] = df["Heart Disease"].map({
            0: "No",
            1: "Yes"
        })
        df["Stroke Rate (%)"] = df["stroke_rate"] * 100
        st.bar_chart(
            df,
            x="Heart Disease",
            y="Stroke Rate (%)"
        )

    # BMI + GLUCOSE
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### BMI Category")
        df = pd.DataFrame(stats["Bmi_group_vs_stroke"])
        df["Stroke Rate (%)"] = df["stroke_rate"] * 100
        st.bar_chart(
            df,
            x="Bmi Category",
            y="Stroke Rate (%)"
        )

    with col2:
        st.markdown("### Glucose Category")
        df = pd.DataFrame(stats["glucose_group_vs_stroke"])
        df["Stroke Rate (%)"] = df["stroke_rate"] * 100
        st.bar_chart(
            df,
            x="Glucose Category",
            y="Stroke Rate (%)"
        )

    
    # SMOKING
    st.markdown("### Smoking Status")
    df = pd.DataFrame(stats["smoking_status_vs_stroke"])
    df["Stroke Rate (%)"] = df["stroke_rate"] * 100
    st.bar_chart(
        df,
        x="Smoking Status",
        y="Stroke Rate (%)"
    )

    # KEY INSIGHT

    st.divider()
    st.markdown("### 🚩 Highest Stroke Rate")
    
    all_groups = []
    feature_data = {
        "Age Group": stats["age_group_vs_stroke"],
        "Hypertension": stats["hypertension_vs_stroke"],
        "Heart Disease": stats["heart_disease_vs_stroke"],
        "BMI Category": stats["Bmi_group_vs_stroke"],
        "Glucose Category": stats["glucose_group_vs_stroke"],
        "Smoking Status": stats["smoking_status_vs_stroke"],
    }

    for feature, records in feature_data.items():
        for record in records:
            category = record[next(
                    key for key in record
                    if key != "stroke_rate")
            ]

            all_groups.append({
                "Feature": feature,
                "Category": category,
                "Stroke Rate": record["stroke_rate"]
            })

    insight_df = pd.DataFrame(all_groups)
    
    highest = insight_df.loc[
        insight_df["Stroke Rate"].idxmax()
    ]

    st.metric(
        label=f"{highest['Feature']} — {highest['Category']}",
        value=f"{highest['Stroke Rate'] * 100:.2f}%"
    )

    st.caption(
        f"The highest observed stroke rate among all categories "
        f"is {highest['Stroke Rate'] * 100:.2f}% "
        f"for {highest['Category']} in {highest['Feature']}."
    )
            
# TAB 2 — Prediction

with tab2:

    options = get_options()
    models = get_models()
    ranges = options["ranges"]

    # LEFT = INPUT
    # RIGHT = RESULT
    col_form, col_result = st.columns([1, 1],gap="large")

    # LEFT: PATIENT INFORMATION
    with col_form:
        
        st.subheader("Patient information")
        with st.form("prediction_form"):
            
            # Personal details
            st.markdown("##### Personal details")
            full_name = st.text_input("Full name")
            
            # Medical information
            st.markdown("##### Medical information")

            age = st.number_input(
                "Age",
                min_value=float(ranges["Age"]["min"]),
                max_value=float(ranges["Age"]["max"]),
                value=float(ranges["Age"]["default"]),
                step=1.0,
                format="%.0f",
            )

            bmi = st.number_input(
                "BMI",
                min_value=float(ranges["Bmi"]["min"]),
                max_value=float(ranges["Bmi"]["max"]),
                value=float(ranges["Bmi"]["default"]),
                step=0.1,
                format="%.1f",
            )

            glucose = st.number_input(
                "Avg Glucose Level",
                min_value=float(ranges["Avg Glucose Level"]["min"]),
                max_value=float(ranges["Avg Glucose Level"]["max"]),
                value=float(ranges["Avg Glucose Level"]["default"]),
                step=0.5,
                format="%.1f",
            )

            hypertension = st.radio("Hypertension?", ["No", "Yes"], horizontal=True,)

            heart_disease = st.radio("Heart disease?",["No", "Yes"],horizontal=True,)

            smoking_status = st.selectbox("Smoking status", options["smoking_status"],)

            ever_married = st.selectbox("Ever married?", options["ever_married"],)

            work_type = st.selectbox("Work type",options["work_type"],)

            # Model
            st.markdown("##### Prediction model")

            model_labels = {
                m["key"]: m["display_name"]
                for m in models
            }

            model_key = st.selectbox(
                "Choose a prediction model",
                list(model_labels.keys()),
                format_func = lambda k: model_labels[k],
                index = 0,
            )

            # Predict
            submitted = st.form_submit_button(
                "🔍 Predict stroke risk",
                use_container_width = True,
            )


    # RIGHT: PREDICTION RESULT
    with col_result:

        st.subheader("Prediction result")
        if not submitted:
            st.markdown(
                """
                <div class='sg-placeholder'>
                    Enter patient information on the left
                    and click <b>Predict stroke risk</b>
                    to see the result here.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            payload = {
                "model": model_key,
                "age": float(age),
                "hypertension": (1 if hypertension == "Yes" else 0),
                "heart_disease": (1 if heart_disease == "Yes" else 0),
                "ever_married": ever_married,
                "work_type": work_type,
                "avg_glucose_level": float(glucose),
                "bmi": float(bmi),
                "smoking_status": smoking_status,
            }

            try:
                result = call_predict(payload)
                
            except requests.exceptions.RequestException as e:
                st.error(
                    f"Prediction request failed: {e}"
                )

            else:
                patient_label = (
                    full_name.strip()
                    if full_name.strip()
                    else "Patient"
                )

                is_stroke = result["prediction"] == 1

                result_class = (
                    "sg-result-yes" if is_stroke
                    else "sg-result-no"
                )

                result_text = (
                    "⚠️ Likely Stroke Risk" if is_stroke
                    else "✅ No Stroke Risk Detected"
                )

                # Patient information
                st.markdown(f"### {patient_label}")
                
                st.markdown(
                    f"**Age:** {int(age)} years old"
                )

                st.markdown(
                    f"**Date of prediction:** "
                    f"{date.today().strftime('%d/%m/%Y')}"
                )

                st.divider()

                # Prediction
                st.markdown(
                    f"<span class='sg-badge {result_class}'>"
                    f"{result_text}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

                st.write("")

                probability = float(result["probability"])

                risk_label = result["risk_label"]

                st.markdown(f"**Probability of stroke:** {probability * 100:.1f}%")

                badge_class = {
                    "Low": "sg-badge-low",
                    "Moderate": "sg-badge-moderate",
                    "High": "sg-badge-high",
                }.get(risk_label,"sg-badge-moderate")

                st.markdown(
                    f"""
                    **Risk level**
                    <span class='sg-badge {badge_class}'> {risk_label}
                    </span>
                    """,
                    unsafe_allow_html=True,
                )

                st.progress(
                    min(max(probability, 0.0), 1.0)
                )

                st.markdown(
                    f"""
                    <div class='sg-card'>
                        <b>Model used:</b>
                        {result["model_display_name"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Disclaimer
                st.markdown(
                    """
                    <div class='sg-disclaimer'>
                        ⚠️ This tool is an educational demo built on a public dataset and is
                        <b>not a medical diagnosis</b>. Please consult a qualified healthcare
                        professional for any real health concern.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )