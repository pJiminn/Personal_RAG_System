from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from dotenv import load_dotenv
from db_init import init_db

# from transformers import Trainer
import shutil
import os
import pdfplumber
import json
import pymysql
import numpy as np
import faiss
from openai import OpenAI
# import torch

import threading
from typing import Any
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#)
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset

from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.storage.blob import BlobServiceClient

# ============================================================
# [0] FastAPI & CORS
# ============================================================
app = FastAPI()

origins = [
    "https://esdl-personal-rag-system-frontend.azurewebsites.net",
    "http://localhost:3000", # 로컬 테스트용
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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

db_host = os.getenv("DB_HOST")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
db_ssl_ca = os.getenv("DB_SSL_CA")

conn = pymysql.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_name,
    port=3306,
    ssl={"ca": db_ssl_ca},
    # cursorclass=pymysql.cursors.DictCursor,  # 결과를 dict로 받기
    charset="utf8mb4",
    autocommit=True,
)

print("✅ MySQL 데이터베이스 연결 성공")

init_db(conn)

def get_cursor():
    return conn.cursor()



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
    cursor = get_cursor()
    try :
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
    finally :
        cursor.close()


# 서버 시작 시 DB → FAISS 로드
load_db_to_faiss()

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
    cursor = get_cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (body.username, body.password),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"user_id": user_id, "username": body.username}
    finally :
        cursor.close()

# @app.post("/login")
# def login(body: LoginRequest):
#     cursor.execute(
#         "SELECT id, password FROM users WHERE username = %s", (body.username,)
#     )
#     row = cursor.fetchone()
#     if not row or row[1] != body.password:
#         raise HTTPException(status_code=401, detail="invalid credentials")
#     return {"user_id": row[0], "username": body.username}

@app.get("/admin/users")
def list_all_users():
    cursor = get_cursor()
    try:
            
        cursor.execute("SELECT id, username FROM users")
        rows = cursor.fetchall()

        return {
            "users": [
                {"id": r[0], "username": r[1]}
                for r in rows
            ]
        }
    finally:
        cursor.close()

# ============================================================
# [7] 프로젝트 CRUD
# ============================================================
class ProjectCreate(BaseModel):
    user_id: int
    name: str
    description: Optional[str] = None
    



@app.post("/projects")
def create_project(body: ProjectCreate):
    cursor = get_cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE id = %s", (body.user_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(status_code=400, detail="Invalid user_id")

        cursor.execute(
            """
            INSERT INTO projects (user_id, name, description, created_at)
            VALUES (%s, %s, %s, NOW())
            """,
            (body.user_id, body.name, body.description),
        )
        conn.commit()

        return {
            "project_id": cursor.lastrowid,
            "name": body.name,
            "description": body.description,
        }
    finally:
        cursor.close()


@app.get("/projects")
def list_projects(user_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            "SELECT id, name, description, created_at FROM projects WHERE user_id = %s",
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
    finally:
        cursor.close()


@app.delete("/projects/{project_id}")
def delete_project(project_id: int):
    cursor = get_cursor()
    try:
        cursor.execute("DELETE FROM projects WHERE id = %s", (project_id,))
        conn.commit()

        # FAISS 메모리에서도 제거
        if project_id in project_indices:
            del project_indices[project_id]
        if project_id in project_documents:
            del project_documents[project_id]

        return {"message": "project deleted"}
    finally:
        cursor.close()


# ============================================================
# [8] 프로젝트별 문서 업로드 & 조회 & 삭제
# ============================================================
@app.post("/projects/{project_id}/upload")
async def upload_file(project_id: int, file: UploadFile = File(...)):
    cursor = get_cursor()
    try:
    # 프로젝트 존재 여부 확인
        cursor.execute("SELECT id FROM projects WHERE id = %s", (project_id,))
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
            INSERT INTO documents (project_id, filename, created_at)
            VALUES (%s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                id = LAST_INSERT_ID(id)
            """,
            (project_id, file.filename),
        )

        conn.commit()

        cursor.execute(
            "SELECT id FROM documents WHERE project_id = %s AND filename = %s",
            (project_id, file.filename),
        )
        doc_row = cursor.fetchone()
        if not doc_row:
            raise HTTPException(status_code=500, detail="failed to create document")
        document_id = doc_row[0]

        # 4. 기존 청크 삭제 후 다시 삽입 (재업로드 대응)
        cursor.execute(
            "DELETE FROM document_chunks WHERE document_id = %s", (document_id,)
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
                VALUES (%s, %s, CAST(%s AS JSON))
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
    finally :
        cursor.close()

@app.get("/projects/{project_id}/documents")
def list_documents(project_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            """
            SELECT id, filename, created_at
            FROM documents
            WHERE project_id = %s
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
    finally:
        cursor.close()


@app.delete("/documents/{document_id}")

def delete_document(document_id: int):
    cursor = get_cursor()
    try:
    # 어떤 프로젝트에 속한 문서인지 확인
        cursor.execute(
            """
            SELECT project_id FROM documents WHERE id = %s
            """,
            (document_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="document not found")

        project_id = row[0]

        # DB 삭제 (ON DELETE CASCADE로 청크 같이 삭제)
        cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
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
            WHERE d.project_id = %s
            """,
            (project_id,),
        )
        rows = cursor.fetchall()

        for chunk_text, emb_json in rows:
            emb = np.array(json.loads(emb_json), dtype="float32").reshape(1, -1)
            idx.add(emb)
            docs.append(chunk_text)

        return {"message": "문서 삭제 및 인덱스 재구성 완료"}
    finally :
        cursor.close()

@app.get("/documents/{document_id}/detail")
def get_document_detail(document_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            "SELECT filename, project_id FROM documents WHERE id = %s",
            (document_id,),
        )
        doc = cursor.fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="not found")

        filename, project_id = doc

        cursor.execute(
            "SELECT chunk FROM document_chunks WHERE document_id = %s",
            (document_id,),
        )
        chunks = [r[0] for r in cursor.fetchall()]

        return {"filename": filename, "project_id": project_id, "chunks": chunks}
    finally:
        cursor.close()

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
    # 1. 초기 연결 및 데이터 확보
    cursor = get_cursor()
    try:
        query = request.query
        project_id = request.project_id
        chat_id = request.chat_id

        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="API 키가 없습니다.")

        # 2. 채팅방 자동 생성 (없을 경우)
        if chat_id is None:
            cursor.execute(
                "INSERT INTO chats (project_id, title, created_at) VALUES (%s, %s, NOW())",
                (project_id, query[:20] if len(query) > 20 else query),
            )
            conn.commit()
            chat_id = cursor.lastrowid

        # 3. 유저 메시지 저장
        cursor.execute(
            "INSERT INTO chat_messages (chat_id, role, content, created_at) VALUES (%s, %s, %s, NOW())",
            (chat_id, "user", query),
        )
        conn.commit()

        # 4. RAG 컨텍스트 준비 (FAISS)
        idx, docs = get_or_create_index(project_id)
        context = ""
        if idx.ntotal > 0:
            emb = get_embedding(query).reshape(1, -1)
            k = min(3, len(docs))
            D, I = idx.search(emb, k)
            context = "\n".join([docs[i] for i in I[0] if i < len(docs)])

        full_prompt = f"문서:\n{context}\n\n질문: {query}"
        print(f"[RAG Prompt]\n{full_prompt}\n{'-'*30}")

        # 5. Generator 함수 정의 (비동기 스트리밍 내부용)
        # 중요: chat_id를 내부에서 안전하게 사용하기 위해 인자로 넘기거나 스코프를 유지합니다.
        def generate(target_chat_id, final_prompt):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": final_prompt}],
                stream=True,
            )

            answer = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    answer += token
                    yield token

            # 스트리밍 종료 후 DB 저장
            c2 = get_cursor()
            try:
                c2.execute(
                    "INSERT INTO chat_messages (chat_id, role, content, created_at) VALUES (%s, %s, %s, NOW())",
                    (target_chat_id, "assistant", answer),
                )
                conn.commit()
            finally:
                c2.close()

        return StreamingResponse(generate(chat_id, full_prompt), media_type="text/plain")

    except Exception as e:
        print(f"Error in chat_stream: {e}")
        return {"error": str(e)}
    finally:
        # 처음 생성한 커서는 여기서 닫아줍니다.
        cursor.close()
    


# ============================================================
# [10] 기존 프론트와의 호환용 (디폴트 프로젝트 1)
# ============================================================

def ensure_default_project() -> int:
    cursor = get_cursor()
    try:
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
            "SELECT id FROM projects WHERE user_id = %s AND name = 'default'",
            (user_id,),
        )
        row = cursor.fetchone()
        if not row:
            cursor.execute(
                """
                INSERT INTO projects (user_id, name, description, created_at)
                VALUES (%s, 'default', '기본 프로젝트', NOW())
                """,
                (user_id,),
            )
            conn.commit()
            project_id = cursor.lastrowid
        else:
            project_id = row[0]

        return project_id
    finally:
        cursor.close()

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
    body = ChatStreamRequest(project_id=DEFAULT_PROJECT_ID, query=query)

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
#         "INSERT INTO chats (project_id, title, created_at) VALUES (%s, %s, datetime('now'))",
#         (project_id, "새 채팅")
#     )
#     conn.commit()

#     return {"chat_id": cursor.lastrowid}

class ChatCreate(BaseModel):
    title: str

@app.post("/projects/{project_id}/chats")
async def create_chat(project_id: int, chat_data: ChatCreate):
    cursor = get_cursor()
    try:
        # DB에 새 채팅방 생성 (SQL문은 본인의 테이블 구조에 맞게 수정하세요)
        cursor.execute(
            "INSERT INTO chats (project_id, title) VALUES (%s, %s)",
            (project_id, chat_data.title)
        )
        # 생성된 ID 가져오기
        new_chat_id = cursor.lastrowid
        # 만약 lastrowid가 안된다면: cursor.execute("SELECT LAST_INSERT_ID()") 사용
        
        return {"chat_id": new_chat_id, "message": "Chat created successfully"}
    except Exception as e:
        print(f"Error creating chat: {e}")
        return {"error": str(e)}, 500
    finally:
        cursor.close()



## 채팅방 목록 불러오기
@app.get("/projects/{project_id}/chats")
async def list_chats(project_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            "SELECT id, title, created_at FROM chats WHERE project_id = %s ORDER BY id DESC",
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
    finally:
        cursor.close()

@app.get("/chats/{chat_id}")
async def get_chat_history(chat_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            "SELECT role, content FROM chat_messages WHERE chat_id = %s ORDER BY id ASC",
            (chat_id,)
        )
        rows = cursor.fetchall()

        return {
            "history": [
                {"role": r[0], "content": r[1]} for r in rows
            ]
        }
    finally :
        cursor.close()




# ---------------------------
# [AUTH] 회원가입
# ---------------------------
@app.post("/auth/register")
async def register(data: Dict[str, str]):
    cursor = get_cursor()
    try: 
        username = data["username"]
        password = data["password"]

        try:
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (%s, %s)",
                (username, password)
            )
            conn.commit()
        except:
            return {"success": False, "message": "이미 존재하는 아이디"}

        return {"success": True}
    
    finally:
        cursor.close()

# ---------------------------
# [AUTH] 로그인
# ---------------------------
@app.post("/auth/login")
async def login(data: Dict[str, str]):
    cursor = get_cursor()
    try:
        username = data["username"]
        password = data["password"]

        # 1) 사용자명으로 먼저 조회
        cursor.execute(
            "SELECT id, username, password FROM users WHERE username=%s",
            (username,)
        )
        user = cursor.fetchone()

        # 디버그 로그
        print(f"[로그인 시도] username: {username}")
        
        if not user:
            print(f"[로그인 실패] 사용자 '{username}'이 DB에 없음")
            return {"success": False, "message": "사용자를 찾을 수 없습니다"}

        user_id, db_username, db_password = user[0], user[1], user[2]
        
        # 2) 비밀번호 확인
        if db_password != password:
            print(f"[로그인 실패] 비밀번호 불일치 (입력: {password}, DB: {db_password})")
            return {"success": False, "message": "비밀번호가 일치하지 않습니다"}

        print(f"[로그인 성공] user_id: {user_id}, username: {db_username}")
        return {
            "success": True,
            "user": {
                "id": user_id,
                "username": db_username
            }
        }
    except Exception as e:
        print(f"[로그인 에러] {str(e)}")
        return {"success": False, "message": f"로그인 중 오류: {str(e)}"}
    finally:
        cursor.close()

class ChatRenameRequest(BaseModel):
    title: str

@app.put("/chats/{chat_id}/rename")
def rename_chat(chat_id: int, body: ChatRenameRequest):
    cursor = get_cursor()
    try:
        cursor.execute(
            "UPDATE chats SET title = %s WHERE id = %s",
            (body.title, chat_id)
        )
        conn.commit()

        return {"success": True}
    finally:
        cursor.close()

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int):
    cursor = get_cursor()
    try:
        cursor.execute(
            "SELECT id FROM chats WHERE id = %s", (chat_id,)
        )
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="chat not found")

        # ✅ 메시지 -> 채팅 순서로 명시 삭제
        cursor.execute("DELETE FROM chat_messages WHERE chat_id = %s", (chat_id,))
        cursor.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
        conn.commit()

        return {"success": True}
    finally:
        cursor.close()

@app.get("/chats/{chat_id}/files")
def get_chat_files(chat_id: int):
    cursor = get_cursor()
    try:
        cursor.execute("""
            SELECT d.id, d.filename
            FROM documents d
            JOIN chat_files cf ON d.id = cf.file_id
            WHERE cf.chat_id = %s
            ORDER BY cf.created_at DESC
        """, (chat_id,))

        files = [
            {"id": r[0], "name": r[1]}
            for r in cursor.fetchall()
        ]


        return {"files": files}
    finally:
        cursor.close()


# ============================================================
# [X] LoRA 학습 상태 전역 관리
# ============================================================
lora_train_status: Dict[int, Dict[str, Any]] = {}

# ============================================================
# [] LoRA 관련 API (deprecated)
# ============================================================

############ 애저 올릴려고 stub응답 처리 
# @app.post("/lora")
# def create_lora(body: LoraCreateRequest):
#     raise HTTPException(
#         status_code=403,
#         detail="LoRA 기능은 현재 배포 환경에서 비활성화되어 있습니다."
#     )
# @app.post("/lora/{lora_id}/train")
# def start_lora_train(lora_id: int):
#     raise HTTPException(
#         status_code=403,
#         detail="LoRA 학습은 GPU 환경에서만 가능합니다."
#     )
# @app.get("/lora/{lora_id}/status", response_model=LoraStatusResponse)
# def get_lora_status(lora_id: int):
#     return LoraStatusResponse(
#         lora_id=lora_id,
#         state="disabled",
#         progress=0,
#         message="LoRA 기능이 비활성화된 배포 환경입니다.",
#         adapter_path=None,
#     )



############################################################################

# ============================================================
# [] LoRA 관련 API (실제 구현)
# ============================================================

### 로라 생성 api
class LoraCreateRequest(BaseModel):
    project_id: int
    name: str
    description: str        # summary / translation / qa / custom
    base_model: str     # mistral / llama 등

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
    print(f"[DEBUG] LoRA {lora_id}] {state} | {progress}% | {message}")

def upload_dataset_to_blob(local_file_path: str, blob_name: str):
    """
    로컬 JSONL 파일을 Azure Blob Storage로 업로드합니다.
    """
    try:
        # 1. 인증 객체 생성 (App Service의 Managed Identity 사용)
        credential = DefaultAzureCredential()
        
        # 2. 서비스 클라이언트 생성
        AZURE_STORAGE_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
        blob_service_client = BlobServiceClient(AZURE_STORAGE_ACCOUNT_URL, credential=credential)
        
        # 3. 컨테이너 지정 (Azure ML은 기본적으로 'azureml-blobstore-...' 컨테이너를 사용함)
        # 직접 만든 컨테이너가 있다면 그 이름을 사용하세요.
        container_name = "training-datasets" 
        
        # 컨테이너가 없으면 생성
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()

        # 4. 파일 업로드
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        print(f"[DEBUG] 업로드 완료: {blob_name}")
        return f"azureml://datastores/workspaceblobstore/paths/{blob_name}" # AI Foundry용 경로 반환

    except Exception as e:
        print(f"[ERROR] 업로드 실패: {e}")
        raise e
    
@app.post("/lora/{lora_id}/train")
def start_lora_train(lora_id: int):
    cursor = get_cursor()
    try:
        # 1. DB에서 정보 조회
        cursor.execute("SELECT base_model, name FROM lora_profiles WHERE id=%s", (lora_id,))
        row = cursor.fetchone()
        if not row: return {"ok": False, "message": "LoRA 정보 없음"}
        base_model = row[0]

        # 2. [중요] 데이터셋 업로드 로직 (Local -> Azure Storage)
        # AI Foundry의 컴퓨팅은 서버의 /data 폴더를 볼 수 없습니다. 
        # 따라서 파일을 스토리지에 업로드해야 합니다.
        local_path = f"data/lora_{lora_id}.jsonl"
        blob_path = f"training-datasets/lora_{lora_id}.jsonl"
        
        upload_dataset_to_blob(local_path, blob_path)
        
        # 3. Azure AI Foundry Job 정의
        job = command(
            code="./train_src",
            command="python train_lora.py --data_path ${{inputs.data}} --base_model ${{inputs.model}} --output_dir ./outputs",
            inputs={
                # path는 실제 업로드된 스토리지 경로여야 합니다.
                "data": Input(type=AssetTypes.URI_FILE, path=f"azureml://datastores/workspaceblobstore/paths/{blob_path}"),
                "model": base_model
            },
            environment="AzureML-pytorch-2.0-ubuntu20.04-py38-cuda11.7-gpu@latest",
            compute="gpu-cluster",
            display_name=f"lora-train-{lora_id}"
        )

        # 4. Job 실행
        returned_job = ml_client.jobs.create_or_update(job)
        azure_job_id = returned_job.name
        
        # 5. [중요] DB에 Job ID 저장
        # 전역 변수(lora_train_status) 대신 DB에 저장해야 서버가 꺼져도 상태를 추적합니다.
        cursor.execute("""
            UPDATE lora_profiles 
            SET status='training', azure_job_id=%s 
            WHERE id=%s
        """, (azure_job_id, lora_id))
        conn.commit()

        return {
            "ok": True, 
            "azure_job_id": azure_job_id,
            "job_url": returned_job.services['Studio'].endpoint
        }
    finally:
        cursor.close()

@app.get("/lora/{lora_id}/status")
def get_lora_status(lora_id: int):
    cursor = get_cursor()
    try:
        # 1. DB에서 azure_job_id 조회
        cursor.execute("SELECT status, azure_job_id FROM lora_profiles WHERE id=%s", (lora_id,))
        row = cursor.fetchone()
        if not row or not row[1]: 
            return {"state": "not_found", "progress": 0}

        status, job_id = row
        
        # 2. Azure ML에서 상태 조회
        job_info = ml_client.jobs.get(job_id)
        azure_status = job_info.status # Running, Completed, Failed, Canceled
        
        # 3. 매핑 및 DB 업데이트 (선택 사항)
        state_map = {
            "Running": "running",
            "Completed": "ready",
            "Failed": "error",
            "Queued": "preparing"
        }
        final_state = state_map.get(azure_status, "training")

        return {
            "lora_id": lora_id,
            "state": final_state,
            "azure_status": azure_status,
            "message": f"Azure Job is {azure_status}"
        }
    finally:
        cursor.close()