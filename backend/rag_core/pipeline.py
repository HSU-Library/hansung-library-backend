# ===== pipeline.py (메인 RAG 파이프라인) =====
import time
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 기존 함수들을 모듈에서 import
from .data_loaders import _load_web_docs, _load_lending_guide, _load_recommend_docs
from .text_processing import _split_docs, _get_or_create_vectorstore
from .retrievers import MetadataAwareCrossEncoderReranker, MetadataAwareRetriever
from .prompts import BOOK_PROMPT, LIBRARY_PROMPT
from .utils import classify_simple, QueryType, validate_book_answer
from .config import EMBED_MODEL

# 전역 초기화 (기존과 동일)
print("\n🚀 인덱스 준비 중...")
web_docs = _load_web_docs()
lending_docs = _load_lending_guide()  # 새로 추가
recommend_docs = _load_recommend_docs()
ALL_DOCS = web_docs + lending_docs + recommend_docs
SPLITS = _split_docs(ALL_DOCS)
print(f"📄 문서 {len(ALL_DOCS)}개 → 청크 {len(SPLITS)}개")

VECTORSTORE = _get_or_create_vectorstore(SPLITS)

print("\n🚀 메타데이터 인식 CrossEncoder Reranker로 업그레이드 중...")

# 메타데이터 인식 Reranker 생성 (기존과 동일)
metadata_aware_reranker = MetadataAwareCrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=3
)

# 리트리버 설정 (기존과 동일)
Book_RETRIEVER = MetadataAwareRetriever(
    vectorstore=VECTORSTORE,
    reranker=metadata_aware_reranker,
    initial_k=15
)

Library_RETRIEVER = VECTORSTORE.as_retriever(
    search_type="mmr", search_kwargs={"k": 5}
)

print("✅ Custom Reranker 파이프라인 완료!")

# LLM 설정 (기존과 동일)
import os
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM = ChatOpenAI(model=MODEL_NAME, temperature=0)

print("🎯 RAG 파이프라인 준비 완료!\n")

# 메인 ask 함수 (기존과 완전 동일)
def ask(question: str, history: List[dict] | None = None) -> Dict[str, Any]:
    """외부에서 호출되는 진입점: 질문을 받아 쿼리 타입별 처리 + 소스 반환"""
    t0 = time.time()

    # 쿼리 분류
    query_type = classify_simple(question)

    # 타입별 리트리버 & 프롬프트 선택
    if query_type == QueryType.BOOK_RECOMMENDATION:
        retriever = Book_RETRIEVER
        prompt = BOOK_PROMPT
    else:
        retriever = Library_RETRIEVER
        prompt = LIBRARY_PROMPT

    retrieved_docs = retriever.invoke(question)

    # Context를 수동으로 구성 (Custom Retriever 사용 시 필요)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    chain = prompt | LLM | StrOutputParser()

    # (2) 답변 생성 - 수동으로 context와 question 전달
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    # 답변 후처리 (할루시네이션 체크)
    if query_type == QueryType.BOOK_RECOMMENDATION:
        answer = validate_book_answer(answer, retrieved_docs)

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

# 테스트 실행 
if __name__ == "__main__":
    question = "4학년 1학기 기술과학 분야에서 추천할 도서는?"
    print(f"📝 질문: {question}")
    
    try:
        result = ask(question)
        print(f"🤖 답변:\n{result['answer']}")
        print(f"⏱️ 응답시간: {result['usage']['latency_sec']}초")
        print(f"📚 검색된 문서 수: {len(result['sources'])}")
    except Exception as e:
        print(f"❌ 오류: {e}")
