# Flask 서버 진입점
# - React 클라이언트와 통신하는 HTTP API 제공
# - /health : 서버 상태 체크
# - /api/chat : 질문을 받아서 RAG 응답 반환
# - /api/books : 책 목록 조회
# - /api/search : 책 검색
# - /api/update_book_status : 스캔 결과 반영 및 상태 갱신

import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
# 내부 서비스 import
from rag_service import ask as rag_ask  # RAG 호출 함수
from services.book_service import (
    load_books,
    save_books,
    generate_expected_barcodes,
    update_book_status_logic
)

# .env 파일 로드 (환경변수 세팅)
load_dotenv()

# Flask App 초기화
app = Flask(__name__)

# CORS 설정: 프론트엔드 개발 서버 도메인 허용
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
origins = [o.strip() for o in allowed.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": origins}}, supports_credentials=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-api")

# ====== 전역 데이터 ======
BOOKS_FILE = os.path.join("data", "books.json")
EXPECTED_BARCODES_FILE = os.path.join("data", "expected_barcodes.json")

# 서버 시작 시 책/expected_barcodes 로드
books = load_books(BOOKS_FILE)
expected_barcodes = generate_expected_barcodes(books, EXPECTED_BARCODES_FILE)

# ====== Health Check ======
@app.get("/health")
def health():
    return {"status": "ok"}, 200

# ====== RAG 챗봇 API ======
@app.post("/api/chat")
def chat():
    """
    챗봇 API 엔드포인트
    요청(JSON): {"message": "...", "history": [...]}
    응답(JSON): {"answer": "...", "sources": [...], "usage": {...}, "requestId": "..."}
    """
    rid = str(uuid.uuid4())  # 요청 식별자
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or data.get("query") or "").strip()
    history = data.get("history") or []

    # 필수 파라미터 확인
    if not question:
        return jsonify({"error": "message (or query) is required", "requestId": rid}), 400

    try:
        # RAG 호출
        res = rag_ask(question, history=history)
        res["requestId"] = rid
        return jsonify(res), 200
    except Exception as e:
        # 에러 발생 시 로깅 후 500 반환
        log.exception("chat error")
        return jsonify({"error": str(e), "requestId": rid}), 500

# ====== Book Management API ======
@app.get("/api/books")
def get_books():
    """책 목록 조회"""
    return jsonify(books), 200

@app.get("/api/search")
def search_books():
    """책 검색 (제목/저자 포함 여부)"""
    query = request.args.get("query", "").lower()
    filtered = [b for b in books if query in b["title"].lower() or query in b["author"].lower()]
    return jsonify(filtered), 200

@app.post("/api/update_book_status")
def update_book_status():
    """
    책 상태 업데이트 API
    요청(JSON): {"location": [...barcode list...]}
    응답(JSON): {available, misplaced, wrong-location, not-available}
    """
    global books, expected_barcodes

    scanned = request.get_json()
    if not scanned:
        return jsonify({"error": "잘못된 데이터"}), 400

    try:
        books, result = update_book_status_logic(books, expected_barcodes, scanned)
        save_books(books, BOOKS_FILE)  # 저장
        return jsonify(result), 200
    except Exception as e:
        log.exception("update_book_status error")
        return jsonify({"error": str(e)}), 500

# ====== 실행 ======
if __name__ == "__main__":
    # 개발 서버 실행 (http://localhost:8000)
    app.run(host="0.0.0.0", port=8000, debug=True)