# Personal RAG System 아키텍처

## 프로젝트 개요

Personal RAG (Retrieval-Augmented Generation) System은 개인 문서 기반 질의응답 및 커스텀 LoRA 모델 학습을 지원하는 풀스택 웹 애플리케이션입니다.

## 시스템 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  - React 19.2.0                                              │
│  - TypeScript                                                │
│  - Tailwind CSS                                              │
│  - Pages: Login, Project Select, Chat, Search                │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
                       │ (CORS Enabled)
┌──────────────────────┴──────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│  - Python 3.10+                                              │
│  - FastAPI Framework                                         │
│  - SQLite Database                                           │
│  - FAISS Vector Search                                       │
│  - OpenAI API Integration                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────┴───┐  ┌───────┴────┐  ┌─────┴────┐
│  SQLite   │  │   FAISS    │  │  OpenAI  │
│ Database  │  │  Indexes   │  │   API    │
│           │  │            │  │          │
│ rag_docs. │  │ In-Memory  │  │ Embedding│
│    db     │  │  Vectors   │  │   & GPT  │
└───────────┘  └────────────┘  └──────────┘
```

## 백엔드 아키텍처 (FastAPI)

### 기술 스택

- **프레임워크**: FastAPI 0.124.0
- **데이터베이스**: SQLite (rag_docs.db)
- **벡터 검색**: FAISS (Facebook AI Similarity Search)
- **임베딩**: OpenAI text-embedding-3-small (1536차원)
- **LLM**: OpenAI GPT (Chat Completion)
- **문서 처리**: pdfplumber (PDF 파싱)
- **모델 학습**: Transformers, PEFT (LoRA)

### 주요 컴포넌트

#### 1. 데이터베이스 스키마

**사용자 및 프로젝트 관리**
- `users`: 사용자 계정 정보
- `projects`: 프로젝트 단위 문서 관리
- `chats`: 채팅 대화방 관리
- `chat_messages`: 채팅 메시지 저장

**문서 및 벡터 관리**
- `documents`: 업로드된 문서 메타데이터
- `document_chunks`: 문서 청크 및 임베딩 벡터 (텍스트 형태로 저장)
- `chat_files`: 채팅별 연결된 문서

**LoRA 모델 관리**
- `lora_profiles`: LoRA 프로필 및 학습 상태

#### 2. FAISS 벡터 검색

- **인덱스 타입**: `IndexFlatIP` (Inner Product, 코사인 유사도)
- **임베딩 차원**: 1536 (OpenAI text-embedding-3-small)
- **프로젝트별 인덱스**: 각 프로젝트마다 독립적인 FAISS 인덱스 관리
- **메모리 기반**: 서버 시작 시 DB에서 FAISS로 로드

#### 3. 주요 API 엔드포인트

**인증 및 사용자 관리**
- `POST /signup`: 회원가입
- `POST /login`: 로그인
- `POST /auth/register`: 회원가입 (대체)
- `POST /auth/login`: 로그인 (대체)
- `GET /admin/users`: 사용자 목록 (관리자)

**프로젝트 관리**
- `POST /projects`: 프로젝트 생성
- `GET /projects`: 프로젝트 목록 조회
- `DELETE /projects/{project_id}`: 프로젝트 삭제

**문서 관리**
- `POST /projects/{project_id}/upload`: 문서 업로드 및 처리
- `GET /projects/{project_id}/documents`: 프로젝트 문서 목록
- `DELETE /documents/{document_id}`: 문서 삭제
- `GET /documents/{document_id}/detail`: 문서 상세 정보

**검색 및 채팅**
- `POST /search`: 벡터 기반 문서 검색
- `POST /chat_stream`: 스트리밍 채팅 (RAG 기반)
- `GET /projects/{project_id}/chats`: 채팅 목록
- `GET /chats/{chat_id}`: 채팅 메시지 조회
- `PUT /chats/{chat_id}/rename`: 채팅 제목 변경
- `DELETE /chats/{chat_id}`: 채팅 삭제
- `GET /chats/{chat_id}/files`: 채팅 연결 문서 목록

**LoRA 모델 관리**
- `POST /lora`: LoRA 프로필 생성
- `GET /lora/list`: LoRA 프로필 목록
- `POST /lora/{lora_id}/train`: LoRA 모델 학습 시작
- `GET /lora/{lora_id}/status`: LoRA 학습 상태 조회
- `GET /lora/{lora_id}/dataset`: LoRA 데이터셋 조회

### 데이터 흐름

#### 문서 업로드 프로세스

```
1. PDF 파일 업로드
   ↓
2. pdfplumber로 텍스트 추출
   ↓
3. 텍스트 청킹 (chunk_size=1000, overlap=200)
   ↓
4. 각 청크를 OpenAI Embedding API로 벡터화
   ↓
5. SQLite DB에 저장 (documents, document_chunks 테이블)
   ↓
6. 프로젝트별 FAISS 인덱스에 추가
```

#### RAG 채팅 프로세스

```
1. 사용자 질의 입력
   ↓
2. 질의를 OpenAI Embedding API로 벡터화
   ↓
3. FAISS 인덱스에서 유사한 문서 청크 검색 (Top-K)
   ↓
4. 검색된 컨텍스트 + 사용자 질의를 프롬프트로 구성
   ↓
5. OpenAI GPT API로 스트리밍 응답 생성
   ↓
6. 메시지 저장 (chat_messages 테이블)
```

#### LoRA 학습 프로세스

```
1. LoRA 프로필 생성
   ↓
2. 학습 데이터셋 자동 생성 (GPT 기반)
   ↓
3. Transformers + PEFT로 LoRA 모델 학습
   ↓
4. 학습 상태를 메모리 및 DB에 저장
   ↓
5. 학습 완료 시 어댑터 경로 저장
```

## 프론트엔드 아키텍처 (Next.js)

### 기술 스택

- **프레임워크**: Next.js 16.0.3 (App Router)
- **언어**: TypeScript 5
- **UI 라이브러리**: React 19.2.0
- **스타일링**: Tailwind CSS 4
- **빌드 도구**: PostCSS

### 페이지 구조

```
app/
├── page.tsx              # 메인 페이지 (채팅 인터페이스)
├── layout.tsx            # 루트 레이아웃
├── LayoutWrapper.tsx     # 레이아웃 래퍼
├── sidebar.tsx           # 사이드바 컴포넌트
├── login/
│   └── page.tsx          # 로그인 페이지
├── project-select/
│   └── page.tsx          # 프로젝트 선택 페이지
├── chat/
│   └── page.tsx          # 채팅 페이지
├── search/
│   └── page.tsx          # 검색 페이지
└── components/
    ├── UploadBox.tsx     # 파일 업로드 컴포넌트
    └── CreateChatModal.tsx  # 채팅 생성 모달
```

### 주요 기능

1. **인증 및 세션 관리**
   - LocalStorage 기반 사용자 인증
   - 자동 로그인 체크

2. **프로젝트 관리**
   - 프로젝트 목록 조회
   - 프로젝트 생성/삭제
   - 프로젝트별 문서 관리

3. **문서 업로드**
   - 드래그 앤 드롭 지원
   - PDF 파일 업로드
   - 업로드 진행 상태 표시

4. **채팅 인터페이스**
   - 실시간 스트리밍 채팅
   - 채팅 히스토리 관리
   - 채팅별 문서 연결

5. **검색 기능**
   - 벡터 기반 문서 검색
   - 검색 결과 표시

## 환경 설정

### 필요한 환경 변수

백엔드 실행을 위해 프로젝트 루트에 `.env` 파일이 필요합니다:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 디렉토리 구조

```
Personal_RAG_System/
├── backend/
│   ├── main.py              # FastAPI 백엔드 메인 파일
│   ├── rag_docs.db          # SQLite 데이터베이스
│   └── uploaded_docs/       # 업로드된 문서 저장소
├── my-chatbot-ui/
│   ├── app/                 # Next.js App Router
│   ├── public/              # 정적 파일
│   └── package.json         # Node.js 의존성
├── venv/                    # Python 가상환경
├── requirements.txt         # Python 의존성 목록
└── .env                     # 환경 변수 (생성 필요)
```

## 기술적 특징

### 1. 멀티 프로젝트 지원
- 사용자별 다중 프로젝트 관리
- 프로젝트별 독립적인 문서 저장소 및 FAISS 인덱스

### 2. 벡터 검색 최적화
- FAISS를 활용한 빠른 유사도 검색
- 프로젝트별 인덱스 분리로 검색 성능 향상

### 3. 스트리밍 응답
- Server-Sent Events (SSE) 기반 스트리밍
- 실시간 채팅 응답 제공

### 4. LoRA 모델 학습
- 커스텀 LoRA 모델 학습 기능
- 비동기 학습 상태 추적

### 5. 확장 가능한 아키텍처
- 모듈화된 코드 구조
- RESTful API 설계
- CORS 지원으로 프론트엔드 분리 배포 가능

## 실행 방법

### 백엔드 실행

```bash
# 가상환경 활성화
.\venv\Scripts\Activate.ps1

# 백엔드 디렉토리로 이동
cd backend

# FastAPI 서버 실행
uvicorn main:app --reload --port 8000
```

### 프론트엔드 실행

```bash
# 프론트엔드 디렉토리로 이동
cd my-chatbot-ui

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 실행
npm run dev
```

백엔드는 `http://localhost:8000`, 프론트엔드는 `http://localhost:3000`에서 실행됩니다.

## 주의사항

1. **API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요
2. **데이터베이스**: SQLite는 개발/소규모 프로덕션용이며, 대규모 배포 시 PostgreSQL 등으로 마이그레이션 권장
3. **FAISS 인덱스**: 현재 메모리 기반이며, 서버 재시작 시 DB에서 재로드됩니다
4. **CORS 설정**: 현재 개발용으로 모든 오리진 허용, 프로덕션 환경에서는 제한 필요

