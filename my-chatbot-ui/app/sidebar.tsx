"use client";

import { useEffect, useState } from "react";

export default function Sidebar({ onSelect }) {
  const [docs, setDocs] = useState([]);

  async function loadDocs() {
    const res = await fetch("http://localhost:8000/documents");
    const data = await res.json();
    setDocs(data.documents || []);
  }

  async function deleteDoc(id: number) {
    await fetch(`http://localhost:8000/documents/${id}`, {
      method: "DELETE",
    });
    loadDocs();
  }

  useEffect(() => {
    loadDocs();
  }, []);

  return (
    <div
      style={{
        width: "260px",
        borderRight: "1px solid #ddd",
        background: "#fff",
        padding: "1rem",
        overflowY: "auto",
      }}
    >
      <h3 style={{ marginBottom: "1rem" }}>📄 문서 목록</h3>

      {docs.length === 0 && <p style={{ color: "#777" }}>문서 없음</p>}

      {docs.map((doc) => (
        <div
          key={doc.id}
          onClick={() => onSelect(doc.id)}   // ⬅️ 문서 클릭 → 모달 열기
          style={{
            padding: "0.7rem",
            background: "#f3f3f3",
            borderRadius: "8px",
            marginBottom: "0.7rem",
            cursor: "pointer",           // 클릭 느낌
          }}
        >
          <div style={{ fontSize: "0.9rem" }}>{doc.filename}</div>

          <button
            onClick={(e) => {
              e.stopPropagation();  // ⬅️ 삭제 버튼 눌러도 모달 안 뜨게
              deleteDoc(doc.id);
            }}
            style={{
              marginTop: "6px",
              width: "100%",
              background: "#ff5c5c",
              border: "none",
              padding: "6px 0",
              borderRadius: "6px",
              color: "white",
              cursor: "pointer",
            }}
          >
            삭제
          </button>
        </div>
      ))}
    </div>
  );
}
