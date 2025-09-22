# rag_core/config.py - 설정 및 상수 분리
import os, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# 임베딩 모델/배치
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = int(os.getenv("OPENAI_EMBED_BATCH", "16"))
TOKEN_CAP_PER_REQUEST = int(os.getenv("OPENAI_EMBED_TOKEN_CAP", "240000"))

# 토큰 길이 함수
_enc = tiktoken.get_encoding("cl100k_base")
def _tok_len(s: str) -> int:
    return len(_enc.encode(s))