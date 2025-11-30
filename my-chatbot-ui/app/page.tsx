"use client";

import { useState } from "react";
import styles from "./ui.module.css";

export default function Home() {
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState("");

  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const backend = "http://localhost:8000";

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    setFileName(file.name);
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${backend}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    alert(data.message);
    setUploading(false);
  }

  async function handleSearch() {
    const res = await fetch(`${backend}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: searchText }),
    });
    const data = await res.json();
    setSearchResults(data.results || []);
  }

  async function handleChat() {
   const userMsg = { role: "user", content: chatInput };
  setChatHistory((prev) => [...prev, userMsg]);

  const res = await fetch("http://localhost:8000/chat_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: chatInput }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let botText = "";
  const botMsg = { role: "assistant", content: "" };
  setChatHistory((prev) => [...prev, botMsg]);

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    botText += chunk;

    // 실시간 업데이트
    setChatHistory((prev) => {
      const updated = [...prev];
      updated[updated.length - 1].content = botText;
      return updated;
    });
  }

  setChatInput("");
  }

  async function handleChatStream() {
  const userMsg = { role: "user", content: chatInput };
  setChatHistory((prev) => [...prev, userMsg]);

  const res = await fetch("http://localhost:8000/chat_stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: chatInput }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  let botText = "";
  const botMsg = { role: "assistant", content: "" };
  setChatHistory((prev) => [...prev, botMsg]);

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    botText += chunk;

    // 실시간 업데이트
    setChatHistory((prev) => {
      const updated = [...prev];
      updated[updated.length - 1].content = botText;
      return updated;
    });
  }

  setChatInput("");
}


  return (
    <div className={styles.container}>
      <h1 className={styles.title}>📚 RAG 챗봇</h1>

      {/* Upload & Search Row */}
      <div className={styles.row}>

        <div className={styles.card}>
          <h3>📄 문서 추가 업로드</h3>
          <input type="file" accept=".pdf" onChange={handleUpload} />
          <p>{fileName ? fileName : "선택된 파일 없음"}</p>
          {uploading && <p>업로드 중...</p>}
        </div>


        <div className={styles.card}>
          <h3>🔍 문서 검색</h3>
          <div className={styles.inputRow}>
            <input
              type="text"
              placeholder="검색어 입력"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              className={styles.input}
            />
            <button onClick={handleSearch} className={styles.btn}>
              검색
            </button>
          </div>

          <div className={styles.searchResults}>
            {searchResults.map((r, i) => (
              <div key={i} className={styles.resultCard}>
                <p className={styles.score}>Score: {r.score}</p>
                <p>{r.text}</p>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Chat Section */}
      <div className={styles.chatBox}>
        <h3>💬 RAG 챗봇</h3>

        <div className={styles.chatWindow}>
          {chatHistory.map((msg, i) => (
            <div
              key={i}
              className={
                msg.role === "user" ? styles.userBubble : styles.botBubble
              }
            >
              {msg.content}
            </div>
          ))}
        </div>

        <div className={styles.chatInputRow}>
          <input
            type="text"
            placeholder="질문 입력"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            className={styles.input}
          />
          <button onClick={handleChat} className={styles.btn}>
            전송
          </button>
        </div>
      </div>
    </div>
  );
}
