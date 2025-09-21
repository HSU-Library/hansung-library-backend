# ===== rag_core/retrievers.py - 리트리버 및 리랭커 =====
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
            'field': None,
            'count': None
        }
        
        # 학년 추출
        grade_match = re.search(r'(\d+)학년', query)
        if grade_match:
            conditions['grade'] = int(grade_match.group(1))
        
        # 학기 추출  
        semester_match = re.search(r'(\d+)학기', query)
        if semester_match:
            conditions['semester'] = int(semester_match.group(1))
        
        # 대출학생수 추출  
        loan_count_match = re.search(r'(\d+)대출학생수', query)
        if loan_count_match:
            conditions['count'] = int(loan_count_match.group(1))
        
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
            score += 500  
            reasons.append(f"학년 일치({conditions['grade']})")
        
        # 학기 매칭 (높은 가중치)
        if conditions['semester'] and metadata.get('학기') == conditions['semester']:
            score += 500  
            reasons.append(f"학기 일치({conditions['semester']})")
            
        # 분야 매칭 (중간 가중치)
        if conditions['field'] and metadata.get('분야') == conditions['field']:
            score += 500
            reasons.append(f"분야 일치({conditions['field']})")

        # 대출학생수: 값이 클수록 가중치 부여
        loan_students = metadata.get('대출학생수')
        if loan_students:
        # 가중치 스케일 조정 가능 (예: *50 → 10명 학생이면 500점 추가)
            score += int(loan_students) * 50  
            reasons.append(f"대출학생수 가중치(+{loan_students*50})")
        
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
            
            from .data_loaders import _extract_title
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