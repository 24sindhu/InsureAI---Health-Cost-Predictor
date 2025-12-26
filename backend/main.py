from fastapi import FastAPI
from backend.schemas import InsuranceInput
from backend.predictor import predict_insurance_cost, get_shap_plot_base64
from backend.explainer import explain_prediction
import io
import base64
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="InsureAI API")


def prepare_df_for_shap(features: dict) -> pd.DataFrame:
    """Prepare scaled dataframe for SHAP"""
    df = pd.DataFrame({
        "age": [features["age"]],
        "bmi": [features["bmi"]],
        "children": [features["children"]],
        "sex_male": [1 if features["sex"] == "male" else 0],
        "smoker_yes": [1 if features["smoker"] == "yes" else 0],
        "region_northwest": [1 if features["region"] == "northwest" else 0],
        "region_southeast": [1 if features["region"] == "southeast" else 0],
        "region_southwest": [1 if features["region"] == "southwest" else 0],
    })
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


@app.post("/predict")
def predict(req: InsuranceInput):
    logger.info("Received prediction request")
    # Prepare features dictionary
    features = {
        "age": req.age,
        "sex": req.sex,
        "bmi": req.bmi,
        "children": req.children,
        "smoker": req.smoker,
        "region": req.region
    }

    # ML prediction
    prediction = predict_insurance_cost(features)

    # AI explanation
    explanation = explain_prediction(features, round(prediction, 2))

    # Generate SHAP plot as base64 image
    try:
        shap_plot_base64 = get_shap_plot_base64(features)
    except Exception:
        logger.exception("SHAP generation failed")
        shap_plot_base64 = None

    logger.info("Prediction completed successfully")
    # Return JSON
    return {
        "prediction": round(prediction, 2),
        "explanation": explanation,
        "shap_plot": shap_plot_base64  # None if not generated
    }