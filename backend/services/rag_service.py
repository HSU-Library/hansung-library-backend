# Flask와 RAG 로직을 연결해주는 얇은 어댑터
# - pipeline.ask()를 호출하고
# - 응답 형식을 Flask API 스펙에 맞게 정규화

from typing import Any, Dict, List
from rag_core.pipeline import ask as core_ask  # 실제 RAG 로직

def ask(question: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """
    질문을 받아 pipeline.ask() 호출 후, 응답을 정리해서 반환
    - answer : 최종 응답 텍스트
    - sources : 참조한 문서/메타데이터
    - usage : 성능 정보 (응답 시간 등)
    """
    result = core_ask(question, history=history)

    # 응답이 단순 문자열이면 dict로 변환
    if isinstance(result, str):
        return {"answer": result, "sources": [], "usage": {}}

    # dict라면 필요한 필드만 추출
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "usage": result.get("usage", {}),
    }