import streamlit as st
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io

@st.cache_data(show_spinner=False)
def call_backend(payload):
    response = requests.post(
        "http://localhost:8000/predict",
        json=payload,
        timeout=30
    )
    return response

def set_theme(theme):
    with open("app/style.css") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.markdown(f"""<script>
        document.body.setAttribute('data-theme', '{theme}');
    </script>""", unsafe_allow_html=True)

# Page setup
st.set_page_config(page_title="Health Insurance Cost Predictor 💸", layout="centered")
# Inject external CSS from style.css
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def display_health_animation(smoker, bmi):
    # Choose animation based on risk factors
    if smoker == "yes":
        anim_url = "https://lottie.host/bde9ccfa-1e40-4c10-bd20-00b4a19b90b7/cS6HiARzai.json"
    elif bmi > 30:
        anim_url = "https://lottie.host/6a53027e-8945-4379-a3e6-caf7062e23c8/YFGLX3Uo1e.json"
    else:
        anim_url = "https://lottie.host/e0f1a489-dbe6-4261-8f3c-91b63c86b89e/pUY0Nq6b0u.json"

    anim = load_lottie_url(anim_url)
    if anim:
        st_lottie(anim, height=220, key="health-status")

# Load Lottie animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_insurance = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_9cyyl8i4.json")

# Show animation
st_lottie(lottie_insurance, height=250, key="insurance")

# Add a thin separator line
st.markdown("""
<div style='border-top: 3px solid #6699cc; margin-top: -30px; margin-bottom: 20px;'></div>
""", unsafe_allow_html=True)

st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)

# Title and description
st.title("🧠 Health Insurance Cost Predictor")
st.markdown("Welcome! Estimate your **annual health insurance cost** based on a few simple inputs.")

st.divider()

# Use sidebar for inputs
# Sidebar for inputs with icons
st.sidebar.header("📋 Fill Your Health Details")

age = st.sidebar.slider("🎂 Age (years)", 18, 64, 30)
sex = st.sidebar.selectbox("⚧️ Sex", ["male", "female"])
bmi = st.sidebar.slider("⚖️ BMI (Body Mass Index)", 15.0, 50.0, 25.0)
children = st.sidebar.selectbox("👶 Number of Children", [0, 1, 2, 3, 4, 5])
smoker = st.sidebar.selectbox("🚬 Smoker?", ["yes", "no"])
region = st.sidebar.selectbox("📍 Region", ["northeast", "northwest", "southeast", "southwest"])
theme_choice = st.sidebar.radio("🌗 Theme", options=["light", "dark"], index=0)
set_theme(theme_choice)

# Input display
st.subheader("🔎 Summary of Your Inputs")
st.write(f"**Age:** {age} | **Sex:** {sex}")
st.write(f"**BMI:** {bmi} | **Children:** {children}")
st.write(f"**Smoker:** {smoker} | **Region:** {region}")

def plot_top_features(features_dict):
    # Sort features by absolute impact (assume explain_prediction returns them in order)
    top_features = list(features_dict.items())
    names = [name for name, val in top_features]
    values = [val for name, val in top_features]

    fig, ax = plt.subplots()
    ax.barh(names, values, color="#2ecc71")
    ax.set_xlabel("Impact on Prediction")
    ax.set_title("Top Contributing Features")
    st.pyplot(fig)

# Prediction
if st.button("🚀 Predict Insurance Cost"):
    payload = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    response = call_backend(payload)

    if response.status_code != 200:
        st.error("Backend error. Please try again.")
        st.stop()

    result = response.json()

    prediction = result.get("prediction")
    explanation = result.get("explanation")
    shap_plot_base64 = result.get("shap_plot")

    if prediction is None or explanation is None:
        st.error("Invalid response from backend.")
        st.stop()

    # Styled box for output
    st.markdown(f"""
    <div style='
        background-color: #e6f4ea;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #2ecc71;
        margin-top: 20px;
        font-size: 1.2rem;
    '>
        💸 <b>Estimated Annual Insurance Cost:</b> <span style='color:#2ecc71'><b>${prediction:,.2f}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Model performance
    with st.expander("📊 Model Performance"):
        st.write("This model was trained using **Linear Regression**.")
        st.write("- **R² Score:** 0.8491 (explains ~85% of variance)")
        st.write("- **RMSE:** $4840.94 (typical error margin)")

    display_health_animation(smoker, bmi)

    # AI explanation
    st.markdown("### 🧠 AI Explanation")
    st.info(explanation)

    # SHAP plot
    if shap_plot_base64:
        st.markdown("### 📈 SHAP Feature Impact")
        shap_bytes = io.BytesIO(base64.b64decode(shap_plot_base64))
        shap_image = Image.open(shap_bytes)
        st.image(shap_image, use_column_width=True)

st.divider()

# Custom Footer Image
with st.expander("📘 About this Project"):
    st.markdown("""
    **Health Insurance Cost Predictor** is a data science web app that estimates a user's annual insurance cost based on their health and demographic information.

    ---
    **🧰 Tech Stack:**
    - [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
    - [![Streamlit](https://img.shields.io/badge/Streamlit-1.32-orange?logo=streamlit)](https://streamlit.io/)
    - [![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)](https://pandas.pydata.org/)
    - [![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikit-learn)](https://scikit-learn.org/)
    - [![LottieFiles](https://img.shields.io/badge/Lottie-Animations-00C3FF?logo=lottie)](https://lottiefiles.com/)

    ---
    **📂 GitHub:**  
    [🔗 View Source Code on GitHub](https://github.com/yourusername/insurance-predictor)
    """)

# Footer
st.markdown("Made with ❤️ by a curious student exploring Data Science.")
st.markdown("[📂 View on GitHub](https://github.com/yourusername/insurance-predictor)")