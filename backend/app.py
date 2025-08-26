# Flask 서버 진입점
# - React 클라이언트와 통신하는 HTTP API 제공
# - /health : 서버 상태 체크
# - /api/chat : 질문을 받아서 RAG 응답 반환

import os
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from rag_service import ask as rag_ask  # RAG 호출 함수 가져오기

# .env 파일 로드 (환경변수 세팅)
load_dotenv()

app = Flask(__name__)

# CORS 설정: 프론트엔드 개발 서버 도메인 허용
allowed = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000")
origins = [o.strip() for o in allowed.split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": origins}}, supports_credentials=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-api")

@app.get("/health")
def health():
    """서버 헬스체크용 엔드포인트"""
    return {"status": "ok"}, 200

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

if __name__ == "__main__":
    # 개발 서버 실행 (http://localhost:8000)
    app.run(host="0.0.0.0", port=8000, debug=True)