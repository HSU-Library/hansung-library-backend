# rag_core/config.py - 설정 및 상수만 분리
from .pipeline import ask

__all__ = ["ask"]

# rag_core/config/__init__.py
from .config import *

# rag_core/data_loaders/__init__.py
from .data_loaders import _load_web_docs, _load_lending_guide, _load_recommend_docs, _extract_title

# rag_core/text_processing/__init__.py
from .text_processing import _split_docs, _get_or_create_vectorstore

# rag_core/retrievers/__init__.py
from .retrievers import MetadataAwareCrossEncoderReranker, MetadataAwareRetriever

# rag_core/prompts/__init__.py
from .prompts import BOOK_PROMPT, LIBRARY_PROMPT

# rag_core/utils/__init__.py
from .utils import classify_simple, QueryType, validate_book_answer
