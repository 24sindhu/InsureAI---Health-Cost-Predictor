# 💸 Health Insurance Cost Predictor

A sleek and interactive Streamlit web app that predicts your **annual health insurance cost** based on factors like age, BMI, region, smoking status, and more — using a trained Linear Regression model.

![Banner](https://cdn.pixabay.com/photo/2017/01/31/18/04/insurance-2021555_960_720.png)

---

## 🚀 Live Demo

Coming soon!  
Or run it locally:

```bash
streamlit run app/streamlit_app.py
✨ Features
💡 Predicts insurance cost using a trained regression model

📊 Interactive UI: sliders, dropdowns, and real-time updates

🌗 Toggle between light and dark themes

🎬 Lottie animations for enhanced UX

🎨 Custom styling via external CSS

📈 R² Score: 0.8491, RMSE: $4840.94

🛠 Tech Stack
Tool	Purpose
Python	Core programming
Pandas	Data manipulation
Scikit-learn	Model training
Streamlit	Web app interface
LottieFiles	UI animations

📦 Folder Structure
css
Copy
Edit
insurance-predictor/ 
├── app/
│   ├── streamlit_app.py         ← Main Streamlit app
│   └── style.css                ← Custom styling
│
├── data/
│   └── insurance.csv            ← Cleaned dataset
│
├── model/
│   ├── model.pkl                ← Trained model
│   └── scaler.pkl               ← Preprocessing scaler
│
├── notebook/
│   └── insurance_analysis.ipynb  ← (your training & exploration notebook)
│
├── README.md                    ← This file
├── .gitignore                   ← Git exclusions
└── requirements.txt             ← Python dependencies
📊 Dataset
Medical Cost Personal Dataset - Kaggle