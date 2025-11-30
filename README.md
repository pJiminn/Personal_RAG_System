## 파일 관리용 DB 스키마

#### 파일 테이블 (파일 단위 관리)
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE,
    created_at TEXT
);

#### 청크 테이블 (벡터 검색용 청크 저장)
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    chunk TEXT,
    embedding TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
