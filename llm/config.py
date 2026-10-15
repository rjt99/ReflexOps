import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter ─────────────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY  = ""
# OPENROUTER_MODEL    = "qwen/qwen3-next-80b-a3b-instruct:free"
OPENROUTER_MODEL    = "openai/gpt-oss-20b:free"
FALLBACK_MODEL      = "openai/gpt-oss-20b:free"

# ── HTTP headers sent with every request ──────────────────────────────────────
DEFAULT_HEADERS = {
    "HTTP-Referer": "https://reflexops.local",
    "X-Title":      "ReflexOps",
}

# ── LLM generation params ─────────────────────────────────────────────────────
LLM_MAX_TOKENS  = 480
LLM_TEMPERATURE = 0.25
LLM_TIMEOUT     = 45          # seconds

# ── Rate-limit retry ──────────────────────────────────────────────────────────
RETRY_MAX_WAIT  = 35          # seconds cap on Retry-After header value

# ── Agent pipeline ─────────────────────────────────────────────────────────────
ACTIVE_AGENTS = ["Detect", "Correlate", "Diagnose", "Recommend"]
