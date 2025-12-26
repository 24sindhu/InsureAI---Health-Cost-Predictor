import subprocess
import sys
import os

def retrain_model():
    """Optional: retrain model from train_model.py"""
    train_file = "backend/train_model.py"
    if os.path.exists(train_file):
        print("Retraining model...")
        subprocess.run([sys.executable, train_file])
    else:
        print("train_model.py not found, skipping retrain.")

def run_backend():
    subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"])
    print("Backend started at http://127.0.0.1:8000")

def run_frontend():
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
    print("Frontend started at http://localhost:8501")

def main():
    retrain_model()
    run_backend()
    run_frontend()

if __name__ == "__main__":
    main()