from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file (if you're using one)
load_dotenv()

# === Path Settings ===
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / '01_raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / '02_processed'

MODEL_BASE_PATH = BASE_DIR / 'models'

# === Model Settings ===
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# === General API Settings ===
REQUEST_TIMEOUT = 10  # seconds
ENABLE_LOGGING = True