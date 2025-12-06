import os
from dotenv import load_dotenv

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("GOOGLE_API_KEY="):
                    k = line.strip().split("=", 1)[1]
                    os.environ["GOOGLE_API_KEY"] = k.strip().strip('"').strip("'")
    except: pass

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL_DETECTION = "models/gemini-2.0-flash" 
MODEL_RESTORATION = "models/gemini-2.0-flash"

MAX_WORKERS = 8
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"