import pickle
import pandas as pd
import shap
import base64
import matplotlib
matplotlib.use("Agg")  # Prevent GUI/Tkinter warnings
import matplotlib.pyplot as plt
from io import BytesIO

# Paths (adjust if needed)
MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"

# Load model and scaler once
model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# Feature order used during training
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "children",
    "sex_male",
    "smoker_yes",
    "region_northwest",
    "region_southeast",
    "region_southwest"
]

# SHAP explainer
background = pd.DataFrame([[40, 25.0, 1, 1, 0, 0, 0, 1]], columns=FEATURE_COLUMNS)
shap_explainer = shap.Explainer(model, background)

def prepare_input(data: dict) -> pd.DataFrame:
    """Convert raw input to model-ready DataFrame"""
    df = pd.DataFrame([{
        "age": data["age"],
        "bmi": data["bmi"],
        "children": data["children"],
        "sex_male": 1 if data["sex"] == "male" else 0,
        "smoker_yes": 1 if data["smoker"] == "yes" else 0,
        "region_northwest": 1 if data["region"] == "northwest" else 0,
        "region_southeast": 1 if data["region"] == "southeast" else 0,
        "region_southwest": 1 if data["region"] == "southwest" else 0,
    }])
    return df[FEATURE_COLUMNS]

def predict_insurance_cost(data: dict) -> float:
    df = prepare_input(data)
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    return round(float(prediction), 2)

def get_shap_plot_base64(data: dict) -> str:
    df = prepare_input(data)
    df_scaled = scaler.transform(df)
    shap_values = shap_explainer(df_scaled)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")