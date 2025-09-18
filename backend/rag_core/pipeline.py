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

from bs4 import BeautifulSoup  
import requests
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

def _extract_title(text: str) -> str:
    if not text:
        return ""
    # 케이스 A: [서명]제목 | [대출…] 패턴
    m = re.search(r"\[서명\]\s*([^|\n]+)", text)
    if m:
        return m.group(1).strip()
    # 케이스 B: "제목 | 대출…" 형태
    if " | " in text:
        return text.split(" | ", 1)[0].strip()
    # 케이스 C: 그냥 한 줄 제목만 있는 경우
    return text.strip()

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
        vs = Chroma(
            embedding_function=embedding,
            persist_directory=str(CHROMA_DIR)
        )
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

# === 메타데이터 인식 Custom Reranker 업그레이드 ===

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple
import re
import time

class MetadataAwareCrossEncoderReranker:
    """
    메타데이터(학년, 학기, 분야) 조건을 우선하는 하이브리드 Reranker
    - CrossEncoder + 메타데이터 매칭 점수 결합
    - 조건 일치를 최우선으로 처리
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 3):
        """
        Args: 
            model_name: CrossEncoder 모델 이름
            top_k: 상위 몇 개 문서를 반환할지
        """
        print(f"🔄 메타데이터 인식 CrossEncoder 모델 로딩 중: {model_name}")
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        print(f"✅ 메타데이터 인식 CrossEncoder 로딩 완료 (top_k={top_k})")

    def extract_conditions(self, query: str) -> Dict[str, Optional[int]]:
        """질문에서 학년, 학기, 분야 조건 추출"""
        conditions = {
            'grade': None,
            'semester': None,
            'field': None
        }
        
        # 학년 추출
        grade_match = re.search(r'(\d+)학년', query)
        if grade_match:
            conditions['grade'] = int(grade_match.group(1))
        
        # 학기 추출  
        semester_match = re.search(r'(\d+)학기', query)
        if semester_match:
            conditions['semester'] = int(semester_match.group(1))
        
        # 분야 추출
        field_keywords = {
            '총류': '총류',
            '철학': '철학', 
            '종교': '종교',
            '사회과학': '사회과학',
            '자연과학': '자연과학',
            '기술과학': '기술과학',
            '예술': '예술',
            '언어': '언어',
            '문학': '문학',
            '역사': '역사'
        }
        
        for keyword, field in field_keywords.items():
            if keyword in query:
                conditions['field'] = field
                break
        
        return conditions

    def calculate_metadata_score(self, metadata: dict, conditions: dict) -> Tuple[int, list]:
        """메타데이터 매칭 점수 계산"""
        score = 0
        reasons = []
        
        # 학년 매칭 (높은 가중치)
        if conditions['grade'] and metadata.get('학년') == conditions['grade']:
            score += 1000  
            reasons.append(f"학년 일치({conditions['grade']})")
        
        # 학기 매칭 (높은 가중치)
        if conditions['semester'] and metadata.get('학기') == conditions['semester']:
            score += 1000  
            reasons.append(f"학기 일치({conditions['semester']})")
            
        # 분야 매칭 (중간 가중치)
        if conditions['field'] and metadata.get('분야') == conditions['field']:
            score += 1200
            reasons.append(f"분야 일치({conditions['field']})")
        
        return score, reasons

    def rerank_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        메타데이터 우선 + CrossEncoder 하이브리드 리랭킹
        """
        if not documents:
            return documents
        
        if len(documents) <= self.top_k:
            return documents

        # 조건 추출
        conditions = self.extract_conditions(query)
        print(f"🎯 추출된 조건: 학년={conditions['grade']}, 학기={conditions['semester']}, 분야={conditions['field']}")
        
        # 쿼리-문서 쌍 생성
        pairs = []
        for doc in documents:
            content = doc.page_content[:512] # 512자 제한
            pairs.append((query, content))

        # CrossEncoder 점수 계산
        print(f"🔄 {len(pairs)}개 문서 하이브리드 리랭킹 중...")
        cross_scores = self.model.predict(pairs)

        # 하이브리드 점수 계산
        scored_docs = []
        
        for i, doc in enumerate(documents):
            cross_score = float(cross_scores[i])
            metadata_score, reasons = self.calculate_metadata_score(doc.metadata, conditions)
            
            # 최종 점수 = 메타데이터 점수 (우선) + CrossEncoder 점수 (보조)
            final_score = metadata_score + cross_score
            
            title = _extract_title(doc.page_content)
            
            scored_docs.append({
                'document': doc,
                'final_score': final_score,
                'metadata_score': metadata_score,
                'cross_score': cross_score,
                'reasons': reasons,
                'title': title
            })

        # 최종 점수로 정렬
        scored_docs.sort(key=lambda x: x['final_score'], reverse=True)

        print(f" ✅ 하이브리드 리랭킹 완료: {len(documents)} -> {self.top_k}개 문서")

        # 디버깅: 상위 문서들의 점수 출력
        for i, item in enumerate(scored_docs[:self.top_k]):
            reasons_str = ', '.join(item['reasons']) if item['reasons'] else '조건 불일치'
            print(f" {i+1}. {item['title'][:40]}...")
            print(f"    최종점수: {item['final_score']:.1f} = 메타데이터({item['metadata_score']}) + CrossEncoder({item['cross_score']:.3f})")
            print(f"    매칭: {reasons_str}")

        # 상위 k개 문서만 반환
        top_docs = [item['document'] for item in scored_docs[:self.top_k]]
        
        return top_docs
    
class MetadataAwareRetriever:
    """
    메타데이터 인식 Reranker가 적용된 Retriever
    """

    def __init__(self, vectorstore, reranker: MetadataAwareCrossEncoderReranker, initial_k: int = 15):
        """초기 검색을 더 많이 해서 누락 방지"""
        self.base_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": initial_k}
        )
        self.reranker = reranker
        self.initial_k = initial_k
        print(f"✅ 메타데이터 인식 Retriever 설정 완료 (초기 검색: {initial_k}개)")
    
    def invoke(self, query: str) -> List[Document]:
        """메타데이터 우선 검색 + 리랭킹"""
        # 1단계: 더 많은 문서 검색 (누락 방지)
        initial_docs = self.base_retriever.invoke(query)

        # 2단계: 메타데이터 + CrossEncoder 하이브리드 리랭킹
        reranked_docs = self.reranker.rerank_documents(initial_docs, query)

        return reranked_docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Legacy 호환성"""
        return self.invoke(query)
    
# === 기존 Custom Reranker를 메타데이터 인식 버전으로 교체 ===
print("\n🚀 메타데이터 인식 CrossEncoder Reranker로 업그레이드 중...")

# 메타데이터 인식 Reranker 생성
metadata_aware_reranker = MetadataAwareCrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=3
)

# 책 추천 관련 RETRIEVER를 Custom Reranker로 교체
Book_RETRIEVER = MetadataAwareRetriever(
    vectorstore=VECTORSTORE,
    reranker=metadata_aware_reranker,
    initial_k=15
)

# 도서관 규정 관련 리트리버 (의미추론만 적용된 기본 상태)
Library_RETRIEVER = VECTORSTORE.as_retriever(
    search_type="mmr", search_kwargs={"k": 5}
)

print("✅ Custom Reranker 파이프라인 완료!")


# 쿼리 분류 (사용자의 질문 분리하기)
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

# 프롬프트 & LLM 준비
from langchain_core.prompts import PromptTemplate

BOOK_PROMPT = PromptTemplate.from_template(
        """
당신은 한성대학교 학술정보관 도서 추천 전문 챗봇입니다.

🚨 **원칙:**
1. 오직 아래 Context에 있는 도서만 활용하세요
2. Context 밖의 도서는 절대 생성하지 마세요
3. 도서명은 반드시 Context에서 정확히 복사하세요

**Context (관련성 점수 순):**
{context}

**질문:** {question}

**출력 형식:**

📚 **추천 도서**: [도서명]

🎯 **추천 이유**: 해당 학년·학기·분야에서 학생들에게 가장 인기 있는 도서입니다.

📖 **도서 소개**: [해당 도서의 주요 내용이나 특징을 2-3문장으로 설명. 도서명을 바탕으로 자유롭고 창의적이게 설명 제공]

🎓 **학습 효과**: [이 책을 통해 얻을 수 있는 지식/스킬을 구체적으로]

⭐ **추천 대상**: [어떤 학생들에게 특히 유용한지 )]

💡 **참고**: 컴퓨터공학부 학생들의 실제 10년치 대출 데이터를 기반으로 한 추천입니다.
"""
)

LIBRARY_PROMPT = PromptTemplate.from_template(
    """
당신은 한성대학교 학술정보관 안내 챗봇입니다.

아래 정보를 참고하여 친절하게 답변해주세요.

**참고 정보:**
{context}

**질문:** {question}

**답변:**
"""
)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM = ChatOpenAI(model=MODEL_NAME, temperature=0)

print("🎯 RAG 파이프라인 준비 완료!\n")
import re

def validate_book_answer(answer: str, retrieved_docs: list) -> str:
    """단일 책 추천 답변 검증 및 교정"""
    
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


# ------------------- 공개 함수 -------------------
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