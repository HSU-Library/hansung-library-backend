# ============================================
# RAG 핵심 파이프라인 (Chroma 버전)
# - 문서 로드 (웹 + CSV)
# - 청크 분할
# - 임베딩 & 벡터스토어 (Chroma persist)
# - 리트리버 & 프롬프트 & LLM 체인 구성
# - ask() 함수로 외부에 제공
# ============================================

from __future__ import annotations
import os, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import bs4
from dotenv import load_dotenv

# LangChain 관련 모듈
from langchain import hub
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import tiktoken # 토큰 길이 계산에 사용

# .env 로드 (OPENAI_API_KEY, OPENAI_MODEL 등 환경 변수 사용)
load_dotenv()

# ------------------- 경로 설정 -------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"  # Chroma persist 디렉터리

# 최근 N년치만 사용 (기본: 최근 5년 → 2021년부터)
FROM_YEAR = int(os.getenv("RAG_FROM_YEAR", datetime.now().year - 4))

# 임베딩 모델/배치
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_BATCH = int(os.getenv("OPENAI_EMBED_BATCH", "16"))  # 한 번에 보내는 문서 수(작게!)
# 안전 토큰 상한(요청당 300k 제한보다 여유 있게)
TOKEN_CAP_PER_REQUEST = int(os.getenv("OPENAI_EMBED_TOKEN_CAP", "240000"))

# 토큰 길이 함수 (text-embedding-3-*와 호환되는 cl100k_base)
_enc = tiktoken.get_encoding("cl100k_base")
def _tok_len(s: str) -> int:
    return len(_enc.encode(s))

# ------------------- 데이터 로딩 -------------------
def _load_web_docs() -> List[Document]:
    """한성대 도서관 웹 규정 문서를 h4+ul 단위로 로드"""
    url = "https://hsel.hansung.ac.kr/intro_data.mir"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    rule_div = soup.find("div", id="intro_rule")
    if not rule_div:
        return []

    docs: List[Document] = []

    # h4 (섹션 제목) + ul (내용 리스트) 추출
    for h4 in rule_div.find_all("h4", class_="sub_title"):
        section_title = h4.get_text(strip=True)
        ul = h4.find_next_sibling("ul")
        if ul:
            items = [li.get_text(strip=True) for li in ul.find_all("li")]
            section_text = section_title + " " + " ".join(items)

            docs.append(Document(
                page_content=section_text,
                metadata={"source": "도서관_규정", "section": section_title}
            ))

    return docs

def _load_recommend_docs() -> List[Document]:
    """추천 결과 CSV(recommend_all.csv)를 로드하여 Document 리스트 반환"""
    csv = DATA_DIR / "recommend_all.csv"
    df = pd.read_csv(csv, encoding="utf-8")

    docs: List[Document] = []
    for _, row in df.iterrows():
        text = (
            f"[서명]{row['서명']} | "
            f"[대출횟수]{row['대출횟수']} | "
            f"[대출학생수]{row['대출학생수']} | "
            f"[학년]{row['학년']} | "
            f"[학기]{row['학기']} | "
            f"[분야]{row['분야']}"
        )
        docs.append(Document(page_content=text, metadata={
            "학년": row["학년"],
            "학기": row["학기"],
            "분야": row["분야"]
        }))
    return docs
# ------------------- 청크 분할(토큰 기준) -------------------
def _split_docs(all_docs: List[Document]) -> List[Document]:
    """토큰 기준으로 안정 분할: 요청당 토큰 초과 방지에 핵심"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # ★ 400~800 사이 권장
        chunk_overlap=60,     # ★ 10% 정도
        length_function=_tok_len,
    )
    return splitter.split_documents(all_docs)

# ------------------- 안전 배치 업로드 유틸 -------------------
def _batch_indices(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]

def _build_index_in_safe_batches(vs: Chroma, docs: List[Document], embedding: OpenAIEmbeddings):
    """
    Chroma 컬렉션에 문서를 '안전 배치'로 나눠 추가.
    - 배치 크기(문서 수) + 배치 내 토큰 총합을 동시에 제한
    - OpenAI 임베딩 API의 요청당 300k 토큰 제한 회피
    """
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    n = len(texts)
    print(f"📦 안전 배치 업로드: 총 {n} 청크")

    start = 0
    while start < n:
        # 1차: 문서 수 기준으로 자르기
        end = min(start + EMBED_BATCH, n)

        # 2차: 토큰 상한 고려해서 end 조정
        tok_sum = 0
        safe_end = start
        for i in range(start, end):
            t = texts[i]
            tok = _tok_len(t)
            if tok_sum + tok > TOKEN_CAP_PER_REQUEST:
                break
            tok_sum += tok
            safe_end = i + 1

        # 만약 한 문서가 너무 길어 cap을 바로 넘는다면, 그 문서를 더 잘게 쪼개야 함
        if safe_end == start:
            # 비정상적으로 긴 청크가 있다면 하드 컷 (마지막 안전장치)
            t = texts[start]
            # 앞쪽 3000 토큰 정도만 임베딩(필요하면 더 작게)
            hard_cut = 3000
            truncated = _enc.decode(_enc.encode(t)[:hard_cut])
            vs.add_texts([truncated], metadatas=[metadatas[start]])
            print(f"⚠️ 긴 청크 하드컷 처리 (index {start}, ~{hard_cut} tok)")
            start += 1
            continue

        # 안전 구간 추가
        vs.add_texts(texts[start:safe_end], metadatas=metadatas[start:safe_end])
        print(f"  → 업로드 {start}:{safe_end} (≈{tok_sum} tokens)")
        start = safe_end

# ------------------- 벡터스토어 준비 -------------------
def _get_or_create_vectorstore(splits: List[Document]) -> Chroma:
    embedding = OpenAIEmbeddings(model=EMBED_MODEL)

    if not CHROMA_DIR.exists():
        print("🔨 새 Chroma 인덱스 생성 (API 호출은 이번 1회)...")
        # 빈 컬렉션을 만들고 → 안전 배치로 add_texts
        vs = Chroma(embedding_function=embedding,
                     persist_directory=str(CHROMA_DIR))
        _build_index_in_safe_batches(
            vs, splits, embedding)
        vs.persist()
        print("✅ 인덱스 생성 & 저장 완료")
    else:
        print("📁 기존 Chroma 인덱스 로드 (API 호출 없음)")
        vs = Chroma(embedding_function=embedding,
                     persist_directory=str(CHROMA_DIR))
    return vs

# ------------------- 전역 초기화 -------------------
print("\n🚀 인덱스 준비 중...")
web_docs = _load_web_docs()
recommend_docs = _load_recommend_docs()
ALL_DOCS = web_docs + recommend_docs
SPLITS = _split_docs(ALL_DOCS)
print(f"📄 문서 {len(ALL_DOCS)}개 → 청크 {len(SPLITS)}개")

VECTORSTORE = _get_or_create_vectorstore(SPLITS)

# Retriever 준비
RETRIEVER = VECTORSTORE.as_retriever(search_type="mmr", search_kwargs={"k": 10})


# 프롬프트 & LLM 준비
try:
    PROMPT = hub.pull("rlm/rag-prompt")
except Exception:
    from langchain_core.prompts import PromptTemplate
    PROMPT = PromptTemplate.from_template(
        "다음 컨텍스트를 바탕으로 답하세요.\n\n{context}\n\n질문: {question}"
    )

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM = ChatOpenAI(model=MODEL_NAME, temperature=0)

RAG_CHAIN = (
    {"context": RETRIEVER, "question": RunnablePassthrough()}
    | PROMPT
    | LLM
    | StrOutputParser()
)

print("🎯 RAG 파이프라인 준비 완료!\n")

# ------------------- 공개 함수 -------------------
def ask(question: str, history: List[dict] | None = None) -> Dict[str, Any]:
    """외부에서 호출되는 진입점: 질문을 받아 답변 + 소스 반환"""
    t0 = time.time()

    # (1) 관련 문서 리트리브
    retrieved_docs = RETRIEVER.invoke(question)

    # (2) 답변 생성
    answer = RAG_CHAIN.invoke(question)

    # (3) 소스 문서 일부 추출
    sources = []
    for d in (retrieved_docs or [])[:5]:
        sources.append({
        "학년": d.metadata.get("학년"),
        "학기": d.metadata.get("학기"),
        "분야": d.metadata.get("분야"),
        "preview": d.page_content[:120] + "..."
    })

    elapsed = time.time() - t0
    return {"answer": answer, "sources": sources, "usage": {"latency_sec": round(elapsed, 2)}}