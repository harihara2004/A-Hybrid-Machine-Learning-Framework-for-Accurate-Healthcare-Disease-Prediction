"""
========================================================================
HYBRID HEALTHCARE DISEASE PREDICTION SYSTEM - STREAMLIT FRONTEND
================================================================================
This app loads the pre-trained Hybrid Voting Model (hybrid_model.pkl),
the fitted StandardScaler (scaler.pkl), and the selected features
(selected_features.pkl) that were produced by hybrid_model.py.

Run this app with:
    streamlit run app.py

Make sure hybrid_model.pkl, scaler.pkl, and selected_features.pkl
are in the SAME FOLDER as this app.py file.
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================
import streamlit as st              # Web app framework
import pandas as pd                  # Data handling
import numpy as np                   # Numerical operations
import pickle                        # Loading saved model/scaler/features
import matplotlib.pyplot as plt      # Plotting
import matplotlib
matplotlib.use("Agg")                # Safe backend for Streamlit rendering


# ==============================================================================
# SECTION 2: PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Hybrid Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# SECTION 3: CUSTOM CSS FOR A PROFESSIONAL LOOK
# ==============================================================================
st.markdown("""
<style>
    /* Overall app background */
    .main {
        background-color: #f5f7fa;
    }

    /* Title styling */
    .title-text {
        font-size: 42px;
        font-weight: 800;
        color: #b91c1c;
        text-align: center;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 16px;
        color: #4b5563;
        text-align: center;
        margin-top: 0px;
        margin-bottom: 25px;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #1f2937;
        border-left: 5px solid #b91c1c;
        padding-left: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    .risk-low {
        background-color: #dcfce7;
        border: 2px solid #16a34a;
        color: #14532d;
    }
    .risk-moderate {
        background-color: #fef9c3;
        border: 2px solid #ca8a04;
        color: #713f12;
    }
    .risk-high {
        background-color: #fee2e2;
        border: 2px solid #dc2626;
        color: #7f1d1d;
    }
    .risk-percentage {
        font-size: 46px;
        font-weight: 800;
        margin: 5px 0px;
    }
    .risk-label {
        font-size: 20px;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #b91c1c;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #991b1b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: LOAD SAVED MODEL, SCALER, AND SELECTED FEATURES
# ==============================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads the trained hybrid model, scaler, and selected feature list
    from the pickle files. Cached so it only loads once per session.
    """
    with open("hybrid_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("selected_features.pkl", "rb") as f:
        selected_features = pickle.load(f)

    return model, scaler, selected_features


# Try loading the artifacts; show a friendly error if files are missing
try:
    hybrid_model, scaler, selected_features = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False


# ==============================================================================
# SECTION 5: FULL LIST OF ORIGINAL 13 FEATURES (ORDER USED DURING TRAINING)
# ==============================================================================
ALL_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


# ==============================================================================
# SECTION 6: HEADER
# ==============================================================================
st.markdown('<p class="title-text">❤️ Hybrid Healthcare Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-powered Heart Disease Risk Assessment using a Weighted Soft-Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting)</p>', unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(
        "⚠️ Could not find `hybrid_model.pkl`, `scaler.pkl`, or `selected_features.pkl`. "
        "Please run `hybrid_model.py` first to train and save the model, and make sure "
        "these three files are in the same folder as this app."
    )
    st.stop()

st.success("✅ Trained Hybrid Model loaded successfully. Enter patient details below to get a prediction.")


# ==============================================================================
# SECTION 7: SIDEBAR - PROJECT INFO
# ==============================================================================
with st.sidebar:
    st.markdown("## 🩺 About This System")
    st.write(
        "This tool predicts the likelihood of Heart Disease using a "
        "**Hybrid Ensemble Machine Learning Model** combining:"
    )
    st.markdown("- Logistic Regression (weight = 1)")
    st.markdown("- Random Forest Classifier (weight = 2)")
    st.markdown("- Gradient Boosting Classifier (weight = 3)")
    st.markdown("---")
    st.markdown("### 📊 Risk Categories")
    st.markdown("🟢 **Low Risk** — 0% to 39%")
    st.markdown("🟡 **Moderate Risk** — 40% to 69%")
    st.markdown("🔴 **High Risk** — 70% to 100%")
    st.markdown("---")
    st.caption("⚠️ Disclaimer: This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")


# ==============================================================================
# SECTION 8: PATIENT INPUT FORM (ALL 13 HEART DISEASE FEATURES)
# ==============================================================================
st.markdown('<p class="section-header">🧾 Enter Patient Details</p>', unsafe_allow_html=True)

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                ("Typical Angina", 0), ("Atypical Angina", 1),
                ("Non-Anginal Pain", 2), ("Asymptomatic", 3)
            ],
            format_func=lambda x: x[0]
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
        thal = st.selectbox(
            "Thalassemia",
            options=[
                ("Normal", 0), ("Fixed Defect", 1),
                ("Reversible Defect", 2), ("Not Described", 3)
            ],
            format_func=lambda x: x[0]
        )

    with col2:
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[("Normal", 0), ("ST-T Abnormality", 1), ("LV Hypertrophy", 2)],
            format_func=lambda x: x[0]
        )
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
        ca = st.selectbox("Number of Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])

    with col3:
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
            format_func=lambda x: x[0]
        )
        st.write("")  # spacing
        st.write("")  # spacing

    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")


# ==============================================================================
# SECTION 9: PREDICTION LOGIC
# ==============================================================================
def predict_patient(patient_dict, model, scaler, selected_features, all_feature_names):
    """
    Takes a dictionary of raw patient inputs, scales them using the saved
    scaler, selects the RFE-selected features, and returns the model's
    prediction and probability of heart disease.
    """
    # Build DataFrame with columns in the exact order used during training
    patient_df = pd.DataFrame([patient_dict])[all_feature_names]

    # Apply the same StandardScaler fitted during training
    patient_scaled = scaler.transform(patient_df)
    patient_scaled_df = pd.DataFrame(patient_scaled, columns=all_feature_names)

    # Keep only the features selected by RFE during training
    patient_selected = patient_scaled_df[selected_features]

    # Predict class label (0 = No Disease, 1 = Disease)
    prediction = model.predict(patient_selected)[0]

    # Predict probability of the positive class (Disease)
    probability = model.predict_proba(patient_selected)[0][1]

    return prediction, probability


def get_risk_category(probability):
    """
    Converts a probability (0-1) into a percentage and risk category.
    """
    risk_percentage = probability * 100

    if risk_percentage < 40:
        return risk_percentage, "Low Risk", "risk-low", "🟢"
    elif risk_percentage < 70:
        return risk_percentage, "Moderate Risk", "risk-moderate", "🟡"
    else:
        return risk_percentage, "High Risk", "risk-high", "🔴"


# ==============================================================================
# SECTION 10: DISPLAY RESULTS AFTER FORM SUBMISSION
# ==============================================================================
if submitted:
    # Extract raw values from selectbox tuples where needed
    patient_input = {
        'age': age,
        'sex': sex[1],
        'cp': cp[1],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs[1],
        'restecg': restecg[1],
        'thalach': thalach,
        'exang': exang[1],
        'oldpeak': oldpeak,
        'slope': slope[1],
        'ca': ca,
        'thal': thal[1]
    }

    # Run prediction using the loaded hybrid model
    prediction, probability = predict_patient(
        patient_input, hybrid_model, scaler, selected_features, ALL_FEATURES
    )

    risk_percentage, risk_category, risk_css_class, risk_emoji = get_risk_category(probability)

    st.markdown('<p class="section-header">📈 Prediction Result</p>', unsafe_allow_html=True)

    result_col1, result_col2 = st.columns([1, 1])

    # ---- Result Card ----
    with result_col1:
        final_label = "⚠️ Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease Detected"

        st.markdown(f"""
        <div class="result-card {risk_css_class}">
            <div class="risk-label">{risk_emoji} {risk_category}</div>
            <div class="risk-percentage">{risk_percentage:.2f}%</div>
            <div class="risk-label">Disease Risk Percentage</div>
            <hr>
            <div class="risk-label">{final_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Gauge-style Bar Chart ----
    with result_col2:
        fig, ax = plt.subplots(figsize=(5, 1.5))

        # Background bar representing 0-100%
        ax.barh([0], [100], color="#e5e7eb", height=0.5)

        # Colored bar representing actual risk percentage
        if risk_percentage < 40:
            bar_color = "#16a34a"
        elif risk_percentage < 70:
            bar_color = "#ca8a04"
        else:
            bar_color = "#dc2626"

        ax.barh([0], [risk_percentage], color=bar_color, height=0.5)

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 40, 70, 100])
        ax.set_xlabel("Risk Percentage (%)")
        ax.set_title("Risk Gauge", fontsize=12, fontweight="bold")

        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

        st.pyplot(fig)

        st.metric(label="Model Confidence (Probability of Disease)", value=f"{probability:.2%}")

    # ---- Patient Summary Table ----
    st.markdown('<p class="section-header">📋 Patient Summary</p>', unsafe_allow_html=True)
    summary_df = pd.DataFrame([patient_input])
    st.dataframe(summary_df, use_container_width=True)

else:
    st.info("👆 Fill in the patient details above and click **Predict Heart Disease Risk** to see results.")


# ==============================================================================
# SECTION 11: FOOTER
# ==============================================================================
st.markdown("---")
st.caption("Hybrid Healthcare Disease Prediction System | Built with Streamlit, scikit-learn & a Weighted Soft-Voting Ensemble")
