import subprocess
import sys

def run_frontend():
    """Runs the Streamlit frontend app."""
    print("Starting frontend at http://localhost:8501...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])

if __name__ == "__main__":
    run_frontend()