import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: autonomous-etl-agent/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
OUTPUT_DIR = DATA_DIR / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# if not GOOGLE_API_KEY:
#     raise RuntimeError("GOOGLE_API_KEY is missing in .env")

# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# OLLAMA_BASE_URL = os.getenv(
#     "OLLAMA_BASE_URL",
#     "http://localhost:11434"
# )

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "gemini-2.5-flash")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")