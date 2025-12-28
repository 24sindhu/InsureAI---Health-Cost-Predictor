import logging
import ollama

logger = logging.getLogger(__name__)

MODEL_NAME = "phi:latest"

def explain_prediction(features: dict, prediction: float) -> str:
    """Call Ollama Phi model for AI explanation"""
    prompt = f"""You are an insurance expert AI.
A user has these details:
Age: {features['age']}, Sex: {features['sex']}, BMI: {features['bmi']:.1f}, 
Children: {features['children']}, Smoker: {features['smoker']}, Region: {features['region']}.

The ML model predicts an annual insurance cost of ${prediction:,.2f}.

Explain in 2-3 sentences why this prediction makes sense, focusing on the key factors affecting the cost."""

    try:
        # Use ollama library's chat function (correct way)
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )
        
        # Extract the message content
        explanation = response['message']['content'].strip()
        logger.info(f"AI explanation generated successfully")
        return explanation
        
    except Exception as e:
        logger.exception(f"Ollama AI call failed: {str(e)}")
        return f"AI explanation unavailable. Predicted cost: ${prediction:,.2f} based on age, BMI, smoking status, and other factors."