import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY: print("WARN: No API Key found. AI features will fail, falling back to local."); 
# Updated to Gemini 3 as requested
MODEL_DETECTION = "models/gemini-3-pro-preview"
MODEL_RESTORATION = "models/gemini-3-pro-image-preview"
MAX_WORKERS = 20
INPUT_ZIP = "zdjecia.zip"
OUTPUT_DIR = "odnowione_final"
TEMP_DIR = "temp_input"