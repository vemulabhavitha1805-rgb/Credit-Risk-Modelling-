import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account",
    "Credit amount", "Duration", "Purpose",
]

# Set premium page config
st.set_page_config(
    page_title="IntelliRisk | Credit Risk Analyzer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global layout & theme overrides */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        font-family: 'Outfit', sans-serif;
        color: #e6edf3;
    }
    
    [data-testid="stHeader"] {
        background: rgba(13, 17, 23, 0.85);
        backdrop-filter: blur(8px);
    }
    
    /* Elegant Title and Subtitle */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #2563EB 50%, #1D4ED8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Cards containers */
    .premium-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid rgba(240, 246, 252, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #58a6ff;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Beautiful Output Panels */
    .result-panel-good {
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.15) 0%, rgba(46, 160, 67, 0.05) 100%);
        border: 2px solid #2ea043;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 25px rgba(46, 160, 67, 0.25);
    }
    
    .result-panel-bad {
        background: linear-gradient(135deg, rgba(248, 81, 73, 0.15) 0%, rgba(248, 81, 73, 0.05) 100%);
        border: 2px solid #f85149;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 25px rgba(248, 81, 73, 0.25);
    }

    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 12px;
        letter-spacing: -0.5px;
    }

    /* Style the analyze button specifically */
    div.stButton > button {
        background: linear-gradient(135deg, #1f6feb 0%, #0969da 100%);
        color: white;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(31, 111, 235, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }

    div.stButton > button:hover {
        box-shadow: 0 6px 20px rgba(31, 111, 235, 0.5);
        transform: translateY(-2px);
        background: linear-gradient(135deg, #388bfd 0%, #1f6feb 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders once with caching to keep app fast
@st.cache_resource
def load_assets():
    """Load model artefacts from the directory containing this application."""
    model = joblib.load(BASE_DIR / "random_trees_credit_model.pkl")
    encoders = {
        "Sex": joblib.load(BASE_DIR / "Sex_label_encoder.pkl"),
        "Housing": joblib.load(BASE_DIR / "Housing_label_encoder.pkl"),
        "Saving accounts": joblib.load(BASE_DIR / "Saving accounts_label_encoder.pkl"),
        "Checking account": joblib.load(BASE_DIR / "Checking account_label_encoder.pkl"),
        "Purpose": joblib.load(BASE_DIR / "Purpose_label_encoder.pkl")
    }
    target_encoder = joblib.load(BASE_DIR / "target_encoder.pkl")
    return model, encoders, target_encoder


def encode_value(value, encoder):
    """Encode one UI value, including the missing-value class used at training."""
    if pd.isna(value):
        missing_values = [item for item in encoder.classes_ if pd.isna(item)]
        if not missing_values:
            raise ValueError("This model was not trained with a 'No Account' category.")
        value = missing_values[0]
    return encoder.transform([value])[0]


def prepare_input(values, encoders, model):
    """Create one encoded, correctly ordered row for model inference."""
    frame = pd.DataFrame([values], columns=FEATURES)
    for column, encoder in encoders.items():
        frame[column] = encode_value(frame.at[0, column], encoder)

    expected_features = list(getattr(model, "feature_names_in_", FEATURES))
    if set(expected_features) != set(FEATURES):
        raise ValueError("The saved model expects a different set of input features.")
    return frame.loc[:, expected_features]


def risk_probability(prediction, probabilities, model):
    """Return the predicted class's probability without assuming class order."""
    class_index = list(model.classes_).index(prediction)
    return float(probabilities[class_index])

try:
    model, encoders, target_encoder = load_assets()
except Exception as e:
    st.error(f"Error loading model or encoders. Please verify that all .pkl files are present in the directory. Details: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-title">💳 INTELLIRISK ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Premium Credit Risk Profile Predictive Modeler</div>', unsafe_allow_html=True)

# Define dropdown mapping tables
sex_map = {"Male": "male", "Female": "female"}
housing_map = {"Own": "own", "Rent": "rent", "Free": "free"}
saving_map = {
    "Little": "little",
    "Moderate": "moderate",
    "Quite Rich": "quite rich",
    "Rich": "rich",
    "No Account": np.nan
}
checking_map = {
    "Little": "little",
    "Moderate": "moderate",
    "Rich": "rich",
    "No Account": np.nan
}
purpose_map = {
    "Radio/TV": "radio/TV",
    "Education": "education",
    "Furniture/Equipment": "furniture/equipment",
    "Car": "car",
    "Business": "business",
    "Domestic Appliances": "domestic appliances",
    "Repairs": "repairs",
    "Vacation/Others": "vacation/others"
}

# Main content splits in Columns
left_col, right_col = st.columns([1.8, 1.2], gap="large")

with left_col:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">👤 APPLICANT PROFILE</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (Years)", min_value=18, max_value=80, value=33, step=1)
        sex = st.selectbox("Sex", options=list(sex_map.keys()), index=0)
    with col2:
        job = st.selectbox(
            "Job Category", 
            options=[0, 1, 2, 3], 
            index=2,
            format_func=lambda x: {
                0: "0 - Unskilled (Non-Resident)", 
                1: "1 - Unskilled (Resident)", 
                2: "2 - Skilled Employee", 
                3: "3 - Management / Highly Skilled"
            }[x]
        )
        housing = st.selectbox("Housing Status", options=list(housing_map.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💰 FINANCIAL CREDIT REQUEST</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        credit_amount = st.number_input("Credit Amount Requested (€)", min_value=250, max_value=20000, value=2300, step=100)
        duration = st.slider("Loan Duration (Months)", min_value=4, max_value=72, value=18, step=1)
    with col4:
        saving_account = st.selectbox("Savings Account Balance", options=list(saving_map.keys()), index=0)
        checking_account = st.selectbox("Checking Account Balance", options=list(checking_map.keys()), index=0)
        
    purpose = st.selectbox("Purpose of Credit", options=list(purpose_map.keys()), index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Center-aligned Analyze button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        predict_btn = st.button("RUN RISK ASSESSMENT", use_container_width=True, type="primary")

with right_col:
    st.markdown('<div class="premium-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 ASSESSMENT REALTIME OUTPUT</div>', unsafe_allow_html=True)
    
    if predict_btn:
        input_values = {
            "Age": age,
            "Sex": sex_map[sex],
            "Job": job,
            "Housing": housing_map[housing],
            "Saving accounts": saving_map[saving_account],
            "Checking account": checking_map[checking_account],
            "Credit amount": credit_amount,
            "Duration": duration,
            "Purpose": purpose_map[purpose],
        }

        try:
            df_input = prepare_input(input_values, encoders, model)
            pred = model.predict(df_input)[0]
            probs = model.predict_proba(df_input)[0]
            translated_pred = target_encoder.inverse_transform([pred])[0]
            confidence = risk_probability(pred, probs, model)
        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Unable to assess this profile: {error}")
            st.stop()
        
        # Display Result elegantly
        if translated_pred == "good":
            st.markdown(f"""
            <div class="result-panel-good">
                <span style="font-size: 3rem;">✅</span>
                <div class="result-title" style="color: #2ea043;">LOW RISK</div>
                <p style="color: #8b949e; font-size: 0.95rem; line-height: 1.4;">
                    Applicant demonstrates a healthy credit profile. Model recommends <b>approval</b> under standard terms.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence probability meter
            st.write("")
            st.write(f"**Confidence score (Good Risk)**: `{confidence * 100:.2f}%`")
            st.progress(confidence)
            
        else:
            st.markdown(f"""
            <div class="result-panel-bad">
                <span style="font-size: 3rem;">⚠️</span>
                <div class="result-title" style="color: #f85149;">HIGH RISK</div>
                <p style="color: #8b949e; font-size: 0.95rem; line-height: 1.4;">
                    Applicant shows markers of elevated financial risk. Model recommends <b>rejection</b> or additional security.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence probability meter
            st.write("")
            st.write(f"**Confidence score (Bad Risk)**: `{confidence * 100:.2f}%`")
            st.progress(confidence)
            
        # Summary details inside card
        st.write("")
        st.markdown("---")
        st.markdown("#### Summary Details")
        st.write(f"- **Loan Leverage**: `{credit_amount / duration:.2f} € / month`")
        st.write(f"- **Age Bracket**: `{age} Years`")
        st.write(f"- **Account Coverage**: `Checking: {checking_account} | Savings: {saving_account}`")
        
    else:
        st.markdown("""
        <div style="text-align: center; color: #8b949e; padding: 60px 20px;">
            <span style="font-size: 4rem; display: block; margin-bottom: 20px; color: #30363d;">⏳</span>
            <h4>System Awaiting Assessment</h4>
            <p style="font-size: 0.9rem; font-weight: 300;">Adjust applicant details on the left, then click <b>assessment button</b> to trigger machine learning analytics.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.caption(
    "Educational demonstration only. A model score must not be the sole basis for a real lending decision."
)

# Elegant Footer
st.write("")
st.markdown("""
<div style="text-align: center; color: #30363d; font-size: 0.8rem; margin-top: 50px;">
    IntelliRisk Assessment Platform • Powered by Random Forest Classification • German Credit Dataset Base
</div>
""", unsafe_allow_html=True)
