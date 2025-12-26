import subprocess
import sys

def run_backend():
    """Runs the FastAPI backend server."""
    print("Starting backend at http://127.0.0.1:8000...")
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", "--reload"])

if __name__ == "__main__":
    run_backend()