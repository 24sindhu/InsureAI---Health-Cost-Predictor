import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
import time
from backend import train_model

def main():
    print("🚀 Starting automated pipeline...")

    # 1️⃣ Train the ML model
    print("🔹 Training the model...")
    train_model.train_and_save_model()
    print("✅ Model training completed.\n")

    # 2️⃣ Launch backend
    print("🔹 Launching backend...")
    backend_process = subprocess.Popen([sys.executable, "scripts/run_backend.py"])
    time.sleep(3)  # Wait for backend to start
    print("✅ Backend started at http://127.0.0.1:8000\n")

    # 3️⃣ Launch frontend
    print("🔹 Launching frontend...")
    frontend_process = subprocess.Popen([sys.executable, "scripts/run_frontend.py"])
    print("✅ Frontend started at http://localhost:8501\n")

    print("🚀 Automated pipeline is running!")
    print("⚠️ Note: Make sure 'ollama serve' is running in another terminal for real AI responses.\n")

    # Keep the script running
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down backend and frontend...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()