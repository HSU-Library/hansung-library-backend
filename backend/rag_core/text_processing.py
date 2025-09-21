# ===== rag_core/text_processing.py - 텍스트 처리 =====
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Tuple
from .config import _tok_len, EMBED_BATCH, TOKEN_CAP_PER_REQUEST, EMBED_MODEL, _enc

def _split_docs(all_docs: List[Document]) -> List[Document]:
    """토큰 기준으로 안정 분할: 요청당 토큰 초과 방지에 핵심"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # ★ 400~800 사이 권장
        chunk_overlap=60,     # ★ 10% 정도
        length_function=_tok_len,
    )
    return splitter.split_documents(all_docs)

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

def _get_or_create_vectorstore(splits: List[Document]) -> Chroma:
    from .config import CHROMA_DIR
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
