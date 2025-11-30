"use client";

import { useState } from "react";
import Sidebar from "./sidebar";

export default function LayoutWrapper({ children }) {
  const [open, setOpen] = useState(true);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [docDetail, setDocDetail] = useState(null);

  async function handleSelectDoc(id) {
    const res = await fetch(`http://localhost:8000/documents/${id}/detail`);
    const data = await res.json();
    setDocDetail(data);
    setSelectedDoc(id);
  }

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      
      {/* 사이드바 */}
      {open && <Sidebar onSelect={handleSelectDoc} />}

      {/* 메인 영역 */}
      <div style={{ flex: 1, position: "relative" }}>

        {/* 사이드바 토글 버튼 📄 */}
        <button
          onClick={() => setOpen(!open)}
          style={{
            position: "absolute",
            left: open ? "30px" : "10px",
            top: "10px",
            zIndex: 100,
            background: "#000000ff",
            color: "white",
            border: "none",
            fontSize: "20px",
            padding: "8px 5px",
            borderRadius: "6px",
            cursor: "pointer",
            transition: "left 0.2s",
        
          }}
        >
          📄
        </button>

        <main
          style={{
            padding: "2rem",
            height: "100vh",
            overflowY: "auto",
            background: "#fafafa",
          }}
        >
          {children}
        </main>

        {/* ======================  
              문서 상세 모달  
           ====================== */}
        {docDetail && (
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              width: "100vw",
              height: "100vh",
              background: "rgba(0,0,0,0.45)",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              zIndex: 999,
            }}
          >
            <div
              style={{
                width: "600px",
                background: "white",
                borderRadius: "12px",
                padding: "2rem",
                boxShadow: "0 8px 20px rgba(0,0,0,0.2)",
              }}
            >
              <h2 style={{ marginBottom: "1rem" }}>
                📄 {docDetail.filename}
              </h2>

              <div
                style={{
                  maxHeight: "400px",
                  overflowY: "auto",
                  whiteSpace: "pre-wrap",
                  background: "#f4f4f4",
                  padding: "1rem",
                  borderRadius: "8px",
                }}
              >
                {docDetail.chunk}
              </div>

              <button
                onClick={() => setDocDetail(null)}
                style={{
                  marginTop: "1rem",
                  width: "100%",
                  padding: "10px",
                  background: "#4a5cff",
                  color: "white",
                  border: "none",
                  borderRadius: "8px",
                  cursor: "pointer",
                }}
              >
                닫기
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
