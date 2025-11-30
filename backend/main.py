from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, pdfplumber, json
import numpy as np
from openai import OpenAI
import faiss
from typing import Dict
from dotenv import load_dotenv
import sqlite3

# ---------------------------
# [0] FastAPI & CORS
# ---------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# [1] 환경 변수 & OpenAI
# ---------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------
# [2] 업로드 폴더
# ---------------------------
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# [3] SQLite DB 설정
# ---------------------------
DB_PATH = "rag_docs.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    chunk TEXT,
    embedding TEXT
)
""")
conn.commit()

# ---------------------------
# [4] FAISS 초기화
# ---------------------------
EMBEDDING_DIM = 1536
index = faiss.IndexFlatL2(EMBEDDING_DIM)
documents = []  # FAISS 순서대로 원문 저장

def load_db_to_faiss():
    """DB에 있는 모든 문서를 불러와 FAISS 인덱스 구성"""
    cursor.execute("SELECT chunk, embedding FROM documents")
    rows = cursor.fetchall()
    documents.clear()
    index.reset()
    for chunk_text, emb_json in rows:
        emb = np.array(json.loads(emb_json), dtype="float32").reshape(1, -1)  # reshape 필수
        index.add(emb)
        documents.append(chunk_text)

# 서버 시작 시 DB → FAISS 로드
load_db_to_faiss()

# ---------------------------
# [5] PDF → 텍스트 변환
# ---------------------------
def pdf_to_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---------------------------
# [6] 텍스트 → 임베딩
# ---------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ---------------------------
# [7] 텍스트 청크 분리
# ---------------------------
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ---------------------------
# [8] 업로드 API
# ---------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = pdf_to_text(file_path)
    chunks = chunk_text(text)

    saved_count = 0
    for chunk in chunks:
        emb = get_embedding(chunk).reshape(1, -1)
        index.add(emb)
        documents.append(chunk)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (filename, chunk, embedding) VALUES (?, ?, ?)",
                (file.filename, chunk, json.dumps(emb.flatten().tolist()))
            )
            conn.commit()
            conn.close()
            saved_count += 1
        except Exception as e:
            print("DB 저장 실패:", e)

    return {"message": f"{file.filename} 업로드 완료, 저장된 청크: {saved_count}/{len(chunks)}"}


# ---------------------------
# [9] 문서 검색 API
# ---------------------------
@app.post("/search")
async def search_docs(query: Dict[str, str]):
    search_text = query["query"]
    if len(documents) == 0:
        return {"results": []}

    emb = get_embedding(search_text).reshape(1, -1)
    k = min(5, len(documents))
    D, I = index.search(emb, k=k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx < len(documents):
            results.append({"text": documents[idx], "score": float(dist)})

    return {"results": results}

# ---------------------------
# [10] RAG 챗봇 API
# ---------------------------
from fastapi.responses import StreamingResponse

@app.post("/chat_stream")
async def chat_stream(request: Dict[str, str]):
    query = request["query"]

    emb = get_embedding(query).reshape(1, -1)
    k = min(3, len(documents))
    D, I = index.search(emb, k=k)

    context_texts = [documents[idx] for idx in I[0] if idx < len(documents)]
    context = "\n".join(context_texts)

    prompt = f"아래 문서를 기반으로 답하세요.\n 문서에 없는 내용은 모른다고 대답하시오. \n\n{context}\n\n질문: {query}"

    def generate():
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")



# [11] 문서 목록 조회
@app.get("/documents")
async def list_documents():
    cursor.execute("SELECT id, filename FROM documents")
    docs = [{"id": r[0], "filename": r[1]} for r in cursor.fetchall()]
    return {"documents": docs}

# [12] 문서 삭제
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    # FAISS와 documents 리스트에서 제거
    cursor.execute("SELECT chunk FROM documents WHERE id=?", (doc_id,))
    row = cursor.fetchone()
    if row:
        chunk_text = row[0]
        if chunk_text in documents:
            idx = documents.index(chunk_text)
            del documents[idx]
            # FAISS 재구성
            index.reset()
            cursor.execute("SELECT embedding, chunk FROM documents")
            rows = cursor.fetchall()
            
            documents.clear()
            for emb_json, chunk in rows:
                emb = np.array(json.loads(emb_json), dtype="float32").reshape(1, -1)
                index.add(emb)
                documents.append(chunk)

    cursor.execute("DELETE FROM documents WHERE id=?", (doc_id,))
    conn.commit()
    return {"message": f"문서 {doc_id} 삭제 완료"}



@app.get("/documents/{doc_id}/detail")
async def get_document_detail(doc_id: int):
    cursor.execute("SELECT filename, chunk FROM documents WHERE id=?", (doc_id,))
    row = cursor.fetchone()

    if not row:
        return {"error": "not found"}

    return {
        "filename": row[0],
        "chunk": row[1]
    }
