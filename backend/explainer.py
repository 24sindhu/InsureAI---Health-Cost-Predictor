from ollama import chat
from backend.predictor import get_shap_values

MODEL_NAME = "phi"

def explain_prediction(input_data, prediction):
    # Get top SHAP features
    shap_features = get_shap_values(input_data)
    top_features = ", ".join(list(shap_features.keys())[:3])  # top 3 features

    prompt = (
        f"You are a health insurance analyst.\n"
        f"User profile:\n"
        f"- Age: {input_data['age']}\n"
        f"- Sex: {input_data['sex']}\n"
        f"- BMI: {input_data['bmi']}\n"
        f"- Children: {input_data['children']}\n"
        f"- Smoker: {input_data['smoker']}\n"
        f"- Region: {input_data['region']}\n\n"
        f"Predicted annual insurance cost: ${prediction:.2f}\n\n"
        f"The top 3 contributing features are: {top_features}\n\n"
        f"Explain:\n"
        f"1. Why the cost is high or low\n"
        f"2. How these top features contribute\n"
        f"3. One suggestion to reduce cost"
    )

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print("Ollama error:", e)
        return "AI explanation temporarily unavailable."