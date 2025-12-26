import subprocess
import sys
import os

print("Launching backend and frontend...")

# Get the path to the current Python executable (inside venv)
python_executable = sys.executable

# Backend
subprocess.Popen([python_executable, os.path.join("scripts", "run_backend.py")])

# Frontend
subprocess.Popen([python_executable, os.path.join("scripts", "run_frontend.py")])