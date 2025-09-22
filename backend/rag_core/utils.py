# ===== rag_core/utils.py - 유틸리티 함수들 =====
import re
from enum import Enum

class QueryType(Enum):
    BOOK_RECOMMENDATION = "book"
    LIBRARY_INFO = "library"

def classify_simple(query: str) -> QueryType:
    """간단한 쿼리 분류"""
    book_keywords = ['추천', '학년', '학기', '분야']
    if any(keyword in query for keyword in book_keywords):
        return QueryType.BOOK_RECOMMENDATION
    return QueryType.LIBRARY_INFO

def validate_book_answer(answer: str, retrieved_docs: list) -> str:
    """단일 책 추천 답변 검증 및 교정"""
    from .data_loaders import _extract_title
    
    # 컨텍스트에서 책 제목들 추출
    context_titles = []
    for doc in retrieved_docs:
        title = _extract_title(getattr(doc, "page_content", str(doc)))
        if title and title not in context_titles:
            context_titles.append(title)
    
    print(f"🔍 컨텍스트에서 추출된 책들: {context_titles[:3]}")  # 디버깅용
    
    # 답변에서 실제 컨텍스트 제목이 포함되었는지 확인
    found_title = None
    for title in context_titles:
        if title in answer:
            found_title = title
            break
    
    if found_title:
        # 올바른 책이 포함된 경우 - 답변 그대로 반환하거나 정리
        print(f"✅ 올바른 책 발견: {found_title}")
        return answer
    else:
        # 컨텍스트에 없는 책이 추천된 경우 - 첫 번째 책으로 대체
        if context_titles:
            best_title = context_titles[0]  # 검색 결과 상위 책
            corrected_answer = f"""추천 기준: 질문 주신 해당 학년과 학기에 대출된 내역을 기반으로 가장 많이 대출된 도서를 추천해 드립니다.

추천 도서: {best_title}"""
            print(f"🔧 답변 교정: 컨텍스트 첫 번째 책으로 대체 → {best_title}")
            return corrected_answer
        else:
            return "죄송합니다. 현재 조건에 맞는 적절한 도서를 찾지 못했습니다."
