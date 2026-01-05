from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from dotenv import load_dotenv

from transformers import Trainer
import shutil
import os
import pdfplumber
import json
import sqlite3
import numpy as np
import faiss
from openai import OpenAI
import torch

import threading
from typing import Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


# ============================================================
# [0] FastAPI & CORS
# ============================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# [1] 환경 변수 & OpenAI
# ============================================================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# [2] 업로드 폴더
# ============================================================
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# [3] SQLite DB 설정
# ============================================================
DB_PATH = "rag_docs.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
def get_cursor():
    return conn.cursor()

cursor = get_cursor()

cursor.execute("PRAGMA foreign_keys = ON")

# ----------------- 기본 테이블 생성 -----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT,
    description TEXT,
    persona TEXT,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    title TEXT,
    created_at TEXT,
    lora_id INTEGER,
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    created_at TEXT,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    filename TEXT,
    created_at TEXT,
    UNIQUE(project_id, filename),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    chunk TEXT,
    embedding TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    file_id INTEGER,
    created_at TEXT,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    FOREIGN KEY(file_id) REFERENCES documents(id) ON DELETE CASCADE
)
""")


## 로라 
cursor.execute("""
CREATE TABLE IF NOT EXISTS lora_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    name TEXT,
    description TEXT,
    purpose TEXT,
    status TEXT,
    adapter_path TEXT,
    base_model TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)
""")


conn.commit()

# ============================================================
# [4] FAISS - 프로젝트별 인덱스 관리
# ============================================================
EMBEDDING_DIM = 1536

# 프로젝트별로 따로 인덱스/문서 메모리 관리
project_indices: Dict[int, faiss.IndexFlatIP] = {}
project_documents: Dict[int, List[str]] = {}


def get_or_create_index(project_id: int):
    """프로젝트별 FAISS 인덱스 / 문서 리스트 보장"""
    if project_id not in project_indices:
        project_indices[project_id] = faiss.IndexFlatIP(EMBEDDING_DIM)
        project_documents[project_id] = []
    return project_indices[project_id], project_documents[project_id]


def load_db_to_faiss():
    """DB 전체를 프로젝트별로 FAISS 인덱스로 로드"""
    project_indices.clear()
    project_documents.clear()

    cursor.execute(
        """
        SELECT dc.id, d.project_id, dc.chunk, dc.embedding
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        """
    )
    rows = cursor.fetchall()

    for _chunk_id, project_id, chunk_text, emb_json in rows:
        idx, docs = get_or_create_index(project_id)
        emb = np.array(json.loads(emb_json), dtype="float32").reshape(1, -1)
        idx.add(emb)
        docs.append(chunk_text)

    print("[FAISS] 전체 로드 완료")
    for pid, docs in project_documents.items():
        print(f" - project {pid}: {len(docs)} chunks")


# 서버 시작 시 DB → FAISS 로드
load_db_to_faiss()



# ============================================================
# [X] LoRA 학습 상태 전역 관리
# ============================================================
lora_train_status: Dict[int, Dict[str, Any]] = {}


# ============================================================
# [5] 유틸: PDF → 텍스트, 텍스트 → 임베딩, 청크
# ============================================================
def pdf_to_text(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(response.data[0].embedding, dtype="float32")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ============================================================
# [6] 간단 로그인 / 회원가입 (비밀번호 해시 X, 실험용)
# ============================================================
class SignUpRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/signup")
def signup(body: SignUpRequest):
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (body.username, body.password),
    )
    conn.commit()
    user_id = cursor.lastrowid
    return {"user_id": user_id, "username": body.username}


@app.post("/login")
def login(body: LoginRequest):
    cursor.execute(
        "SELECT id, password FROM users WHERE username = ?", (body.username,)
    )
    row = cursor.fetchone()
    if not row or row[1] != body.password:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return {"user_id": row[0], "username": body.username}


# ============================================================
# [7] 프로젝트 CRUD
# ============================================================
class ProjectCreate(BaseModel):
    user_id: int
    name: str
    description: Optional[str] = None
    


class ProjectCreate(BaseModel):
    user_id: int
    name: str
    description: str
    #persona: str   #  LoRA용 핵심 필드


@app.post("/projects")
def create_project(body: ProjectCreate):

    # ✅ 유저 존재 검사 (FOREIGN KEY 방지)
    cursor.execute("SELECT id FROM users WHERE id = ?", (body.user_id,))
    user = cursor.fetchone()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    cursor.execute(
        """
        INSERT INTO projects (user_id, name, description, created_at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (body.user_id, body.name, body.description),
    )
    conn.commit()

    return {
        "project_id": cursor.lastrowid,
        "name": body.name,
        "description": body.description,
    }


@app.get("/projects")
def list_projects(user_id: int):
    cursor.execute(
        "SELECT id, name, description, created_at FROM projects WHERE user_id = ?",
        (user_id,),
    )
    rows = cursor.fetchall()
    return {
        "projects": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "created_at": r[3],
            }
            for r in rows
        ]
    }


@app.delete("/projects/{project_id}")
def delete_project(project_id: int):
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()

    # FAISS 메모리에서도 제거
    if project_id in project_indices:
        del project_indices[project_id]
    if project_id in project_documents:
        del project_documents[project_id]

    return {"message": "project deleted"}


# ============================================================
# [8] 프로젝트별 문서 업로드 & 조회 & 삭제
# ============================================================
@app.post("/projects/{project_id}/upload")
async def upload_file(project_id: int, file: UploadFile = File(...)):
    # 프로젝트 존재 여부 확인
    cursor.execute("SELECT id FROM projects WHERE id = ?", (project_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="project not found")

    # 1. 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. PDF → Text → Chunk
    text = pdf_to_text(file_path)
    chunks = chunk_text(text)

    # 3. documents 테이블에 등록
    cursor.execute(
        """
        INSERT OR IGNORE INTO documents (project_id, filename, created_at)
        VALUES (?, ?, datetime('now'))
        """,
        (project_id, file.filename),
    )
    conn.commit()

    cursor.execute(
        "SELECT id FROM documents WHERE project_id = ? AND filename = ?",
        (project_id, file.filename),
    )
    doc_row = cursor.fetchone()
    if not doc_row:
        raise HTTPException(status_code=500, detail="failed to create document")
    document_id = doc_row[0]

    # 4. 기존 청크 삭제 후 다시 삽입 (재업로드 대응)
    cursor.execute(
        "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
    )
    conn.commit()

    # 5. 청크 + FAISS 반영
    idx, docs = get_or_create_index(project_id)
    saved_count = 0

    for chunk in chunks:
        emb = get_embedding(chunk)

        cursor.execute(
            """
            INSERT INTO document_chunks (document_id, chunk, embedding)
            VALUES (?, ?, ?)
            """,
            (document_id, chunk, json.dumps(emb.tolist())),
        )

        idx.add(emb.reshape(1, -1))
        docs.append(chunk)
        saved_count += 1

    conn.commit()

    return {
        "message": f"{file.filename} 업로드 완료",
        "document_id": document_id,
        "saved_chunks": saved_count,
        "total_chunks": len(chunks),
    }


@app.get("/projects/{project_id}/documents")
def list_documents(project_id: int):
    cursor.execute(
        """
        SELECT id, filename, created_at
        FROM documents
        WHERE project_id = ?
        ORDER BY id DESC
        """,
        (project_id,),
    )
    rows = cursor.fetchall()
    return {
        "documents": [
            {"id": r[0], "filename": r[1], "created_at": r[2]} for r in rows
        ]
    }


@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    # 어떤 프로젝트에 속한 문서인지 확인
    cursor.execute(
        """
        SELECT project_id FROM documents WHERE id = ?
        """,
        (document_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    project_id = row[0]

    # DB 삭제 (ON DELETE CASCADE로 청크 같이 삭제)
    cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()

    # 해당 프로젝트의 FAISS 인덱스를 다시 구축
    idx, docs = get_or_create_index(project_id)
    idx.reset()
    docs.clear()

    cursor.execute(
        """
        SELECT dc.chunk, dc.embedding
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        WHERE d.project_id = ?
        """,
        (project_id,),
    )
    rows = cursor.fetchall()

    for chunk_text, emb_json in rows:
        emb = np.array(json.loads(emb_json), dtype="float32").reshape(1, -1)
        idx.add(emb)
        docs.append(chunk_text)

    return {"message": "문서 삭제 및 인덱스 재구성 완료"}


@app.get("/documents/{document_id}/detail")
def get_document_detail(document_id: int):
    cursor.execute(
        "SELECT filename, project_id FROM documents WHERE id = ?",
        (document_id,),
    )
    doc = cursor.fetchone()
    if not doc:
        raise HTTPException(status_code=404, detail="not found")

    filename, project_id = doc

    cursor.execute(
        "SELECT chunk FROM document_chunks WHERE document_id = ?",
        (document_id,),
    )
    chunks = [r[0] for r in cursor.fetchall()]

    return {"filename": filename, "project_id": project_id, "chunks": chunks}


# ============================================================
# [9] 프로젝트별 검색 & RAG 챗봇
# ============================================================
class SearchRequest(BaseModel):
    project_id: int
    query: str


@app.post("/search")
def search_docs(body: SearchRequest):
    project_id = body.project_id
    query = body.query

    idx, docs = get_or_create_index(project_id)
    if idx.ntotal == 0 or len(docs) == 0:
        return {"results": []}

    emb = get_embedding(query).reshape(1, -1)
    k = min(5, len(docs))
    D, I = idx.search(emb, k)

    results = []
    for idx_, dist in zip(I[0], D[0]):
        if 0 <= idx_ < len(docs):
            results.append({"text": docs[idx_], "score": float(dist)})

    return {"results": results}

class ChatStreamRequest(BaseModel):
    project_id: int
    chat_id: Optional[int] = None
    query: str


class LoraCreateRequest(BaseModel):
    project_id: int
    name: str
    description: Optional[str] = None
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"


class LoraStatusResponse(BaseModel):
    lora_id: int
    state: str          # "created" | "running" | "done" | "error"
    progress: int       # 0~100
    message: str
    adapter_path: Optional[str] = None



@app.post("/chat_stream")
async def chat_stream(request: ChatStreamRequest):
    try:
        query = request.query
        project_id = request.project_id
        chat_id = request.chat_id

        # OpenAI API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

        # 채팅방 없으면 자동 생성
        if chat_id is None:
            cursor.execute(
                "INSERT INTO chats (project_id, title, created_at) VALUES (?, ?, datetime('now'))",
                (project_id, query[:20] if len(query) > 20 else query),
            )
            conn.commit()
            chat_id = cursor.lastrowid

        # 유저 메시지 저장
        cursor.execute(
            "INSERT INTO chat_messages (chat_id, role, content, created_at) VALUES (?, ?, ?, datetime('now'))",
            (chat_id, "user", query),
        )
        conn.commit()

        # ✅ LoRA 연결 여부 확인
        cursor.execute("SELECT lora_id FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        lora_id = row[0] if row else None

        adapter_path = None
        if lora_id:
            cursor.execute("SELECT adapter_path FROM lora_profiles WHERE id = ?", (lora_id,))
            r2 = cursor.fetchone()
            adapter_path = r2[0] if r2 else None
            # 지금은 adapter_path만 읽어오고, 실제 추론은 OpenAI로.
            # 나중에 로컬 Mistral+LoRA 서버를 붙이면 여기서 분기.

        # 프로젝트별 FAISS 인덱스
        idx, docs = get_or_create_index(project_id)

        context = ""
        if idx.ntotal > 0:
            try:
                emb = get_embedding(query).reshape(1, -1)
                k = min(3, len(docs))
                D, I = idx.search(emb, k)
                context = "\n".join([docs[i] for i in I[0] if i < len(docs)])
            except Exception as e:
                print(f"임베딩 검색 에러: {e}")
                context = ""

        prompt = f"""
아래 문서 기반으로만 답하시오.
문서에 없는 내용이면 "모르겠습니다"라고 대답하십시오.
만약 정보에 관한 대화가 아닌 일상 대화라면, 친구처럼 대답하세요.

문서:
{context}

질문: {query}
"""

        def generate():
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )

                answer = ""

                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        answer += token
                        yield token

                cursor.execute(
                    "INSERT INTO chat_messages (chat_id, role, content, created_at) VALUES (?, ?, ?, datetime('now'))",
                    (chat_id, "assistant", answer),
                )
                conn.commit()
            except Exception as e:
                error_msg = f"OpenAI API 에러: {str(e)}"
                print(error_msg)
                yield f"\n\n[에러 발생: {error_msg}]\n"

        return StreamingResponse(generate(), media_type="text/plain")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"서버 에러: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"채팅 처리 중 에러 발생: {str(e)}")



# ============================================================
# [10] 기존 프론트와의 호환용 (디폴트 프로젝트 1)
# ============================================================

def ensure_default_project() -> int:
    """기존 /upload, /chat_stream 등 쓰던 UI를 살리기 위한 기본 프로젝트"""
    # 1) 기본 유저 생성
    cursor.execute("SELECT id FROM users WHERE username = 'default'")
    row = cursor.fetchone()
    if not row:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES ('default', 'default')"
        )
        conn.commit()
        user_id = cursor.lastrowid
    else:
        user_id = row[0]

    # 2) 기본 프로젝트 생성
    cursor.execute(
        "SELECT id FROM projects WHERE user_id = ? AND name = 'default'",
        (user_id,),
    )
    row = cursor.fetchone()
    if not row:
        cursor.execute(
            """
            INSERT INTO projects (user_id, name, description, created_at)
            VALUES (?, 'default', '기본 프로젝트', datetime('now'))
            """,
            (user_id,),
        )
        conn.commit()
        project_id = cursor.lastrowid
    else:
        project_id = row[0]

    return project_id


DEFAULT_PROJECT_ID = ensure_default_project()

# ↓↓↓ 기존 프론트엔드가 사용하는 /upload 그대로 동작하게 래핑 ↓↓↓


@app.post("/upload_legacy")
async def upload_file_legacy(file: UploadFile = File(...)):
    """기존 /upload 대신 /upload_legacy 사용 (default project)"""
    return await upload_file(DEFAULT_PROJECT_ID, file)


@app.post("/chat_stream_legacy")
async def chat_stream_legacy(request: Dict[str, str]):
    """기존 프론트용: project_id 없이 쓰던 버전"""
    query = request["query"]
    body = ChatRequest(project_id=DEFAULT_PROJECT_ID, query=query)
    return await chat_stream(body)


@app.get("/documents_legacy")
def list_documents_legacy():
    return list_documents(DEFAULT_PROJECT_ID)



# ## 채팅방 생성 API

# class CreateChatRequest(BaseModel):
#     title: str


# @app.post("/projects/{project_id}/chats")
# def create_chat(project_id: int):
#     cursor.execute(
#         "INSERT INTO chats (project_id, title, created_at) VALUES (?, ?, datetime('now'))",
#         (project_id, "새 채팅")
#     )
#     conn.commit()

#     return {"chat_id": cursor.lastrowid}





## 채팅방 목록 불러오기
@app.get("/projects/{project_id}/chats")
async def list_chats(project_id: int):
    cursor = get_cursor()

    cursor.execute(
        "SELECT id, title, created_at FROM chats WHERE project_id = ? ORDER BY id DESC",
        (project_id,)
    )
    rows = cursor.fetchall()
    cursor.close()

    return {
        "chats": [
            {"id": r[0], "title": r[1], "created_at": r[2]}
            for r in rows
        ]
    }

# 
@app.get("/chats/{chat_id}")
async def get_chat_history(chat_id: int):
    cursor.execute(
        "SELECT role, content FROM chat_messages WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,)
    )
    rows = cursor.fetchall()

    return {
        "history": [
            {"role": r[0], "content": r[1]} for r in rows
        ]
    }

@app.post("/login")
async def login(req: Dict[str, str]):
    email = req["email"]

    cursor.execute("SELECT id, email FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()

    if not user:
        cursor.execute(
            "INSERT INTO users (email) VALUES (?)",
            (email,)
        )
        conn.commit()
        user_id = cursor.lastrowid
    else:
        user_id = user[0]

    return {
        "user": {
            "id": user_id,
            "email": email
        }
    }


# ---------------------------
# [AUTH] 회원가입
# ---------------------------
@app.post("/auth/register")
async def register(data: Dict[str, str]):
    username = data["username"]
    password = data["password"]

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
    except:
        return {"success": False, "message": "이미 존재하는 아이디"}

    return {"success": True}


# ---------------------------
# [AUTH] 로그인
# ---------------------------
@app.post("/auth/login")
async def login(data: Dict[str, str]):
    username = data["username"]
    password = data["password"]

    cursor.execute(
        "SELECT id, username FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()

    if not user:
        return {"success": False}

    return {
        "success": True,
        "user": {
            "id": user[0],
            "username": user[1]
        }
    }


class ChatRenameRequest(BaseModel):
    title: str


@app.put("/chats/{chat_id}/rename")
def rename_chat(chat_id: int, body: ChatRenameRequest):
    cursor.execute(
        "UPDATE chats SET title = ? WHERE id = ?",
        (body.title, chat_id)
    )
    conn.commit()

    return {"success": True}


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int):

    cursor.execute(
        "SELECT id FROM chats WHERE id = ?", (chat_id,)
    )
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="chat not found")

    # ✅ 메시지 -> 채팅 순서로 명시 삭제
    cursor.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()

    return {"success": True}


@app.get("/chats/{chat_id}/files")
def get_chat_files(chat_id: int):
    cursor.execute("""
        SELECT d.id, d.filename, d.path
        FROM documents d
        JOIN chat_files cf ON d.id = cf.file_id
        WHERE cf.chat_id = ?
        ORDER BY cf.created_at DESC
    """, (chat_id,))

    files = [
        {"id": r[0], "name": r[1], "path": r[2]}
        for r in cursor.fetchall()
    ]

    return {"files": files}


### 로라 생성 api
class LoraCreateRequest(BaseModel):
    project_id: int
    name: str
    description: str        # summary / translation / qa / custom
    base_model: str     # mistral / llama 등


@app.post("/lora")
def create_lora(body: LoraCreateRequest):
    # 1) DB 저장
    cursor.execute(
        """
        INSERT INTO lora_profiles (project_id, name, description, adapter_path, created_at, base_model)
        VALUES (?, ?, ?, ?, datetime('now'), ?)
        """,
        (body.project_id, body.name, body.description or "", "", body.base_model),
    )
    conn.commit()
    lora_id = cursor.lastrowid

    # 2) 데이터셋 생성 경로
    os.makedirs("data", exist_ok=True)
    dataset_path = f"data/lora_{lora_id}.jsonl"

    # 3) GPT 기반 dataset 생성
    purpose = body.description or ""
    dataset_text = generate_chat_dataset_from_purpose(
        purpose,
        num_samples=30
    )

    # 4) JSONL 파일 저장
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(dataset_text)

    # 5) 상태 업데이트
    update_status(
        lora_id,
        state="created_dataset",
        progress=10,
        message="Chat 메시지 기반 데이터셋 생성 완료"
    )

    return {
        "lora_id": lora_id,
        "dataset_path": dataset_path
    }

### c채팅시 로라 연결
class CreateChatRequest(BaseModel):
    title: str
    lora_id: Optional[int] = None

@app.post("/projects/{project_id}/chats")
def create_chat(project_id: int, body: CreateChatRequest):
    cursor.execute(
        """
        INSERT INTO chats (project_id, title, lora_id, created_at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (project_id, body.title, body.lora_id)
    )
    conn.commit()

    return {"chat_id": cursor.lastrowid}


## 로라 학습 
import threading
import time

def run_lora_training(lora_id: int):
    try:
        cursor.execute("UPDATE lora_profiles SET status='training' WHERE id=?", (lora_id,))
        conn.commit()

        # ✅ 여기에 네 QLoRA 학습 코드가 들어가면 된다
        # subprocess.run(["python", "train_lora.py", "--lora_id", str(lora_id)])

        for i in range(10):
            time.sleep(2)  # 학습 중인 것처럼

        adapter_path = f"./lora_adapters/lora_{lora_id}"

        cursor.execute("""
            UPDATE lora_profiles
            SET status='ready', adapter_path=?
            WHERE id=?
        """, (adapter_path, lora_id))

        conn.commit()

    except:
        cursor.execute("UPDATE lora_profiles SET status='failed' WHERE id=?", (lora_id,))
        conn.commit()


## 학습시작
@app.post("/lora/{lora_id}/train")
def start_lora_train(lora_id: int):
    st = lora_train_status.get(lora_id)

    if not st:
        return {"ok": False, "message": "LoRA 상태 없음"}

    if st.get("state") == "running":
        return {"ok": False, "message": "이미 학습 중"}

    # 데이터셋 체크
    dataset_path = f"data/lora_{lora_id}.jsonl"
    if not os.path.exists(dataset_path):
        return {"ok": False, "message": "데이터셋이 없어 학습 불가"}

    t = threading.Thread(target=train_mistral_lora, args=(lora_id,), daemon=True)
    t.start()

    update_status(lora_id, "running", 20, "학습 준비 중...")

    return {"ok": True, "message": "학습을 시작했습니다."}


def update_status(lora_id, state, progress, message, adapter_path=None):
    # 기존 상태가 있으면 가져오고, 없으면 기본 상태로 초기화
    prev = lora_train_status.get(lora_id, {
        "state": "created",
        "progress": 0,
        "message": "",
        "adapter_path": None,
    })

    # 업데이트
    lora_train_status[lora_id] = {
        "state": state,
        "progress": progress,
        "message": message,
        "adapter_path": adapter_path or prev["adapter_path"]
    }

    # 터미널 디버그용
    print(f"[LoRA {lora_id}] {state} | {progress}% | {message}")



## QLoRa 학습함수
def train_mistral_lora(lora_id: int):
    try:
        update_status(lora_id, "preparing", 20, "데이터셋 로드 중...")

        # 1) 데이터셋 로드
        dataset_path = f"data/lora_{lora_id}.jsonl"
        raw_dataset = load_dataset("json", data_files=dataset_path)["train"]

        update_status(lora_id, "preparing", 30, f"데이터셋 {len(raw_dataset)}개 로드 완료")

        # 2) 모델 로드
        cursor.execute("SELECT base_model FROM lora_profiles WHERE id=?", (lora_id,))
        base_model = cursor.fetchone()[0]

        update_status(lora_id, "preparing", 40, "토크나이저/모델 로드 중...")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )


    
        update_status(lora_id, "preparing", 55, "LoRA 구성 중...")

        # 3) LoRA 적용
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        # 4) 토큰화
        update_status(lora_id, "preparing", 65, "토큰화 진행 중...")

        def format_example(example):
            text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )

            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=1024,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

        # 5) 학습
        update_status(lora_id, "running", 70, "학습 시작")

        output_dir = f"lora_adapters/lora_{lora_id}"
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to=[],
        )

        trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
        trainer.train()

        update_status(lora_id, "saving", 90, "어댑터 저장 중...")
        model.save_pretrained(output_dir)

        # DB 업데이트
        cursor.execute(
            "UPDATE lora_profiles SET adapter_path=?, created_at=datetime('now') WHERE id=?",
            (output_dir, lora_id),
        )
        conn.commit()

        update_status(lora_id, "done", 100, "학습 완료")

    except Exception as e:
        update_status(lora_id, "error", 0, f"에러: {repr(e)}")


# ============================================================
# [LoRA 관리 API]
# ============================================================

# @app.post("/lora")
# def create_lora(body: LoraCreateRequest):
#     cursor.execute(
#         """
#         INSERT INTO lora_profiles (project_id, name, description, adapter_path, created_at, base_model)
#         VALUES (?, ?, ?, ?, datetime('now'), ?)
#         """,
#         (body.project_id, body.name, body.description or "", "", body.base_model),
#     )
#     conn.commit()
#     lora_id = cursor.lastrowid

    # 상태 초기화
    lora_train_status[lora_id] = {
        "state": "created",
        "progress": 0,
        "message": "생성됨(학습 전)",
        "adapter_path": None,
    }

    return {"lora_id": lora_id}


@app.get("/lora/{lora_id}/status", response_model=LoraStatusResponse)
def get_lora_status(lora_id: int):
    st = lora_train_status.get(lora_id, {
        "state": "unknown",
        "progress": 0,
        "message": "상태 정보 없음",
        "adapter_path": None,
    })

    cursor.execute("SELECT adapter_path FROM lora_profiles WHERE id=?", (lora_id,))
    row = cursor.fetchone()
    adapter_path = row[0] if row else None

    return LoraStatusResponse(
        lora_id=lora_id,
        state=st["state"],
        progress=st["progress"],
        message=st["message"],
        adapter_path=adapter_path,
    )


    cursor.execute(
        "SELECT adapter_path FROM lora_profiles WHERE id = ?",
        (lora_id,),
    )
    row = cursor.fetchone()
    adapter_path = row[0] if row else None

    return LoraStatusResponse(
        lora_id=lora_id,
        state=st["state"],
        progress=st["progress"],
        message=st["message"],
        adapter_path=adapter_path,
    )


@app.get("/admin/users")
def list_all_users():
    cursor.execute("SELECT id, username, email FROM users ORDER BY id DESC")
    rows = cursor.fetchall()

    return {
        "users": [
            {"id": r[0], "username": r[1], "email": r[2]}
            for r in rows
        ]
    }

@app.get("/lora/list")
def list_lora(project_id: int):
    cursor.execute("""
        SELECT id, name, description, status, adapter_path
        FROM lora_profiles
        WHERE project_id = ?
        ORDER BY id DESC
    """, (project_id,))

    rows = cursor.fetchall()

    return {
        "loras": [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "status": r[3],
                "adapter_path": r[4],
            }
            for r in rows
        ]
    }


@app.get("/lora/{lora_id}/dataset")
def preview_lora_dataset(lora_id: int, limit: int = 5):
    dataset_path = f"data/lora_{lora_id}.jsonl"

    if not os.path.exists(dataset_path):
        return {"ok": False, "message": "데이터셋 없음"}

    preview = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            preview.append(json.loads(line))

    total = sum(1 for _ in open(dataset_path, "r", encoding="utf-8"))

    return {
        "ok": True,
        "dataset_path": dataset_path,
        "total_samples": total,
        "preview": preview,
    }

##GPT API를 호출해서 LoRA 학습용 messages[] dataset을 만드는 코드
def generate_chat_dataset_from_purpose(purpose: str, num_samples: int = 30):
    prompt = f"""
너는 LoRA 학습용 데이터셋 생성기다.

다음은 사용자가 작성한 LoRA의 목적 설명이다:

[LoRA 목적 설명]
{purpose}

이 목적을 충실하게 반영할 수 있는 최적의 학습 데이터셋을 생성해야 한다.

========================================================
1) 너의 작업(반드시 지켜야 함)

- 사용자가 작성한 목적 설명을 읽고, 그 목적이 어떤 범주에 해당하는지 스스로 판단한다.
- 예:
  - 영어 논문 번역 LoRA → 번역 중심 데이터셋
  - 용어 설명 LoRA → 개념 정의, 예시 설명 중심 데이터셋
  - 논문 요약 LoRA → 원문 → 요약 중심 데이터셋
  - 논문 Q&A LoRA → 질문 → 분석/해석 중심 데이터셋
  - 문서 기반 tutor LoRA → multi-turn 설명 중심 데이터셋
  - 그 외 → 목적에 맞게 구조를 설계

- 판단한 범주에 맞춰 학습 태스크를 자동 생성한다.
- 그리고 그 태스크에 가장 적합한 messages[] 형태의 학습 데이터를 생성한다.

========================================================
2) 출력 데이터 형식

각 라인은 JSONL 형식이어야 하며 다음 구조를 따른다:

{{
  "messages": [
    {{ "role": "system", "content": "<목적을 반영한 assistant의 역할 정의>" }},
    {{ "role": "user", "content": "<사용자의 입력 예시>" }},
    {{ "role": "assistant", "content": "<사용자의 목적을 달성하는 이상적인 응답>" }}
  ]
}}

========================================================
3) 절대적으로 지켜야 할 규칙 (TemplateError 방지용 → 매우 중요)

⚠️ messages 배열은 반드시 다음 규칙을 따라야 한다:

- system은 0~1회만 등장하며 반드시 맨 앞에 온다.
- 그 다음은 user → assistant → user → assistant … 순서로 번갈아야 한다.
- user가 연속으로 두 번 오면 안 된다.
- assistant가 연속으로 두 번 오면 안 된다.
- multi-turn을 사용할 때도 반드시 user/assistant 교대 규칙을 유지해야 한다.

즉, 허용되는 패턴의 예:

- [system], user, assistant
- [system], user, assistant, user, assistant
- [system], user, assistant, user, assistant, user, assistant

절대 허용되지 않는 패턴:

- user, user
- assistant, assistant
- assistant이 user보다 먼저 등장
- system이 중간에 등장하는 경우

========================================================
4) multi-turn 생성 규칙

목적이 복잡한 경우 multi-turn을 포함해도 좋지만,
항상 user → assistant → user → assistant 형태여야 한다.

단일 턴도 가능하다.

========================================================
5) 출력 규칙

- 총 {num_samples}개의 JSON 오브젝트를 JSONL 형식으로 출력한다.
- 각 줄 하나당 하나의 JSON.
- 코드블록은 절대 사용하지 않는다.
- JSON 외의 텍스트는 포함하지 않는다.

========================================================
이제 위 모든 규칙을 지키며 JSONL 데이터셋을 생성하라.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 고품질 LoRA 학습 데이터셋을 자동 생성하는 AI이며, role 순서 규칙을 엄격히 준수해야 한다."},
            {"role": "user", "content": prompt}
        ]
    )

    dataset_text = response.choices[0].message.content
    return dataset_text
