import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY: print("WARN: No API Key"); exit(1)
MODEL_DETECTION = "models/gemini-1.5-pro"
MODEL_RESTORATION = "models/gemini-1.5-pro"
MAX_WORKERS = 20
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"