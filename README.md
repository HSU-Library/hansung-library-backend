# 📚 Hansung Library Backend

한성대학교 도서관을 위한 **챗봇 기반 사서 관리 시스템** 백엔드입니다.  
Flask + RAG(Retrieval-Augmented Generation) + 라즈베리파이 바코드 리더기를 기반으로,  
학생/도서관 이용자는 챗봇을 통해 도서관 규정 및 추천 도서를 확인할 수 있고,  
관리자는 서가 점검 및 책 상태를 실시간으로 관리할 수 있습니다.

---

## 🚀 주요 기능

### 🧑‍🎓 학생/사용자
- 챗봇을 통한 **도서관 규정 Q&A**
- 챗봇을 통한 학과/학년별 **추천 도서 제공**
- React 프론트엔드와 연동 (`/chat` 페이지)

### 📚 사서/관리자
- **도서 목록 조회 및 검색**
- **서가 상태 확인** (책 있음 / 없음 / 순서 오류 / 잘못 배치)
- 라즈베리파이 바코드 리더기를 이용한 **자동 점검**
- 관리자 페이지 (`/admin/books`, `/admin/book-shelf`)와 연동

---

## 📂 프로젝트 구조

```bash
HSU_Library_Backend/
├── backend/
│   ├── app.py                  # Flask 서버 진입점
│   ├── services/
│   │   ├── book_service.py     # 도서 관리 로직
│   │   └── rag_service.py      # RAG 기반 챗봇 서비스
│   ├── rag_core/               # RAG 파이프라인 관련 핵심 모듈
│   │   ├── __init__.py
│   │   ├── config.py           # 환경설정 및 공통 상수 정의
│   │   ├── data_loaders.py     # 데이터 로딩 및 전처리 모듈
│   │   ├── pipeline.py         # RAG 파이프라인 실행 로직
│   │   ├── prompts.py          # 프롬프트 템플릿 관리
│   │   ├── retrievers.py       # 벡터 검색기/리트리버 정의
│   │   ├── text_processing.py  # 텍스트 전처리 유틸
│   │   └── utils.py            # 공통 유틸리티 함수 모음
│   ├── chroma_db/              # Chroma 인덱스 저장소 (RAG 검색용 벡터 DB)
│   └── requirements.txt        # 의존성 패키지 목록
└── notebooks/                  # 데이터 전처리 및 RAG 실험용 주피터 노트북
```
---

## ⚙️ 실행 방법

1. 가상환경 생성 및 활성화
   
```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows (CMD)
.\venv\Scripts\Activate.ps1 # Windows (PowerShell)
```

2. 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경변수 설정 .env 파일 생성
```bash
OPENAI_API_KEY=your_openai_api_key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
FLASK_ENV=development
PORT=8080
```

4. 서버 실행
```bash
python app.py
```
---
## 📌 기술 스택
- Backend: Python, Flask, Flask-CORS, Flask-SocketIO
- AI: LangChain, Chroma, OpenAI API
- Infra: Raspberry Pi (바코드 리더기), JSON 파일 저장
