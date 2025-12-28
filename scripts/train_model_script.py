"""
Automated ML training pipeline for InsureAI
"""

import sys
import os

# Allow imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.train_model import train_and_save_model


def main():
    print("🚀 Starting automated model training...")
    train_and_save_model()
    print("✅ Model training completed and saved.")


if __name__ == "__main__":
    main()