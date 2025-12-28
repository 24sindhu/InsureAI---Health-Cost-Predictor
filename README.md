````markdown
# 💸 Health Insurance Cost Predictor

A sleek and interactive **Streamlit web app** that predicts your annual health insurance cost based on factors like age, BMI, region, smoking status, and more — using a trained **Linear Regression model** with optional **AI explanations** via Ollama Phi.

![Banner](https://cdn.pixabay.com/photo/2017/01/31/18/04/insurance-2021555_960_720.png)

---

## 🚀 Live Demo

Coming soon!  
Or run it locally:

```bash
# Activate your virtual environment
.\venv\Scripts\activate

# Launch the automated pipeline (trains model, starts backend & frontend)
python scripts/auto_run.py
````

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Features

* 💡 **Predicts insurance cost** using a trained Linear Regression model
* 🧠 **AI Explanation:** Uses Ollama Phi to provide natural language insights into predictions
* 📊 **Interactive UI:** sliders, dropdowns, and real-time updates
* 🌗 Toggle between light and dark themes
* 🎬 Lottie animations for enhanced UX
* 🎨 Custom styling via external CSS
* 📈 **Model Performance:**

  * R² Score: 0.7836
  * RMSE: $5,796.28

---

## 🛠 Tech Stack

| Tool         | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Core programming             |
| Pandas       | Data manipulation            |
| Scikit-learn | Model training               |
| Streamlit    | Web app interface            |
| SHAP         | Feature impact visualization |
| Ollama Phi   | AI explanation integration   |

---

## 📦 Folder Structure

```
insurance-predictor/
├── app/
│   ├── streamlit_app.py          ← Main Streamlit app
│   └── style.css                 ← Custom styling
│
├── backend/
│   ├── predictor.py              ← ML prediction functions
│   ├── train_model.py            ← Model training script
│   └── explainer.py              ← Ollama AI integration
│
├── data/
│   └── insurance.csv             ← Dataset
│
├── model/
│   ├── model.pkl                 ← Trained Linear Regression model
│   └── scaler.pkl                ← Feature scaler
│
├── scripts/
│   ├── auto_run.py               ← Automates training + backend + frontend
│   ├── run_all.py                ← Run backend & frontend separately
│   ├── run_backend.py
│   ├── run_frontend.py
│   └── train_model_script.py     ← Standalone training script
│
├── notebook/
│   └── insurance_analysis.ipynb  ← Exploratory data analysis & model training
│
├── README.md                     ← This file
├── .gitignore                    ← Git exclusions
└── requirements.txt              ← Python dependencies
```

---

## 📊 Dataset

Medical Cost Personal Dataset - [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## ⚡ Notes

* Make sure **Ollama is running** locally for AI explanations:

  ```bash
  ollama serve
  ```
* The automated pipeline (`auto_run.py`) will train the model, start the **FastAPI backend**, and launch the **Streamlit frontend** automatically.
* SHAP feature impact visualizations are included for interpretability.

```