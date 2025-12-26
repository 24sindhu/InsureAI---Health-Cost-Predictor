from backend.predictor import get_shap_plot_base64

# Try to import Ollama, but don’t break if it’s not installed
try:
    from ollama import chat
    ollama_available = True
except ImportError:
    ollama_available = False

def explain_prediction(features: dict, prediction: float) -> str:
    """
    Generates AI explanation using LLM if available, otherwise returns
    a simple textual explanation.
    """

    # Generate SHAP plot for visualization
    _ = get_shap_plot_base64(features)

    if ollama_available:
        prompt = f"""
        A health insurance cost prediction model estimated the annual cost as ${prediction}.
        
        User details:
        - Age: {features['age']}
        - BMI: {features['bmi']}
        - Children: {features['children']}
        - Sex: {features['sex']}
        - Smoker: {features['smoker']}
        - Region: {features['region']}

        Explain in simple terms why this cost might be high or low, considering the user's features.
        """

        try:
            response = chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception:
            # If Ollama fails, fallback to simple explanation
            return (
                f"The model predicts an annual insurance cost of ${prediction}.\n"
                "Factors like age, BMI, smoking status, and number of children "
                "influence this prediction."
            )
    else:
        # Ollama not available
        return (
            f"The model predicts an annual insurance cost of ${prediction}.\n"
            "Factors like age, BMI, smoking status, and number of children "
            "influence this prediction."
        )