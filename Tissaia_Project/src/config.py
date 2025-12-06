import os
from dotenv import load_dotenv

load_dotenv()

# Fallback: manually read .env if load_dotenv fails or ENV var not present
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if line.strip().startswith("GOOGLE_API_KEY="):
                        k = line.strip().split("=", 1)[1]
                        os.environ["GOOGLE_API_KEY"] = k.strip().strip('"').strip("'")
    except: pass

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_DETECTION = "models/gemini-3-pro-preview"
MODEL_RESTORATION = "models/gemini-3-pro-image-preview"
MAX_WORKERS = 4
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"

print(f"\n[CONFIG] API Key Present: {'YES' if API_KEY else 'NO'}")
print(f"[CONFIG] Engine: {MODEL_RESTORATION}")
