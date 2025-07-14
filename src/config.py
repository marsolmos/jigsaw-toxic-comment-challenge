from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / '01_raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / '02_processed'