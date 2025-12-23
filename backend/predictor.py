import pickle
import pandas as pd
import shap

MODEL_PATH = "model/model.pkl"
SCALER_PATH = "model/scaler.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

FEATURE_NAMES = [
    "age", "bmi", "children", "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest"
]

def prepare_input(data: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "age": [data["age"]],
        "bmi": [data["bmi"]],
        "children": [data["children"]],
        "sex_male": [1 if data["sex"] == "male" else 0],
        "smoker_yes": [1 if data["smoker"] == "yes" else 0],
        "region_northwest": [1 if data["region"] == "northwest" else 0],
        "region_southeast": [1 if data["region"] == "southeast" else 0],
        "region_southwest": [1 if data["region"] == "southwest" else 0],
    })

def predict_insurance_cost(data: dict) -> float:
    df = prepare_input(data)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    return float(prediction)

def get_shap_values(data: dict) -> dict:
    df = prepare_input(data)
    scaled = scaler.transform(df)

    explainer = shap.Explainer(model, df)
    shap_values = explainer(df)
    
    feature_importances = {}
    for name, value in zip(FEATURE_NAMES, shap_values.values[0]):
        feature_importances[name] = float(value)
    
    sorted_features = dict(sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))
    return sorted_features