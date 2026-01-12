"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import CreateChatModal from "./components/CreateChatModal"; // 🔥 모달 import

export default function Sidebar({
  projectId,
  setProjectId,
  chatId,
  setChatId,
  setChatHistory,
  projectRefreshKey,
}: any) {
  const backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  
  const router = useRouter();

  const [projects, setProjects] = useState<any[]>([]);
  const [chats, setChats] = useState<any[]>([]);
  const [projectFiles, setProjectFiles] = useState<any[]>([]);

  const [showModal, setShowModal] = useState(false);
  const [loraList, setLoraList] = useState<any[]>([]);

  // -----------------------------
  // 프로젝트 목록 로드
  // -----------------------------
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user") || "null");
    if (!user) return;

    fetch(`${backend}/projects?user_id=${user.id}`)
      .then((res) => res.json())
      .then((data) => setProjects(data.projects || []));
  }, []);

  // 프로젝트 자동 갱신
  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("user") || "null");
    if (!user) return;

    fetch(`${backend}/projects?user_id=${user.id}`)
      .then((res) => res.json())
      .then((data) => setProjects(data.projects || []));
  }, [projectRefreshKey]);

  // -----------------------------
  // 채팅 + 파일 + LoRA 목록 로드
  // -----------------------------
  useEffect(() => {
    if (!projectId) return;

    fetch(`${backend}/projects/${projectId}/chats`)
      .then((res) => res.json())
      .then((data) => setChats(data.chats || []));

    fetch(`${backend}/projects/${projectId}/documents`)
      .then((res) => res.json())
      .then((data) => setProjectFiles(data.documents || []));

    // 🔥 LoRA 목록도 가져오기
    fetch(`${backend}/lora/list?project_id=${projectId}`)
      .then((res) => res.json())
      .then((data) => setLoraList(data.loras || []));
  }, [projectId]);

  // -----------------------------
  // ★ 채팅 생성 처리 함수 (모달에서 호출)
  // -----------------------------
  async function handleCreateChat(data: any) {
    // case 1) 일반 채팅
    if (data.lora_id === null && !data.newLora) {
      await fetch(`${backend}/projects/${projectId}/chats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: data.title, lora_id: null }),
      });
    }

    // case 2) 기존 LoRA 연결
    if (data.lora_id && !data.newLora) {
      await fetch(`${backend}/projects/${projectId}/chats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: data.title, lora_id: data.lora_id }),
      });
    }

    // case 3) 새 LoRA 만들기
    if (data.newLora) {
      // 1) LoRA 생성
      const res1 = await fetch(`${backend}/lora`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          name: data.newLora.name,
          description: data.newLora.purpose,
          base_model: data.newLora.base_model,
        }),
      });
      const d1 = await res1.json();
      const newLoraId = d1.lora_id;

      // 2) 학습 시작
      await fetch(`${backend}/lora/${newLoraId}/train`, {
        method: "POST",
      });

      // 3) LoRA가 연결된 채팅 생성
      await fetch(`${backend}/projects/${projectId}/chats`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: data.title, lora_id: newLoraId }),
      });
    }

    // 리스트 갱신
    const res2 = await fetch(`${backend}/projects/${projectId}/chats`);
    const data2 = await res2.json();
    setChats(data2.chats || []);
  }

  return (
    <aside
      style={{
        width: "260px",
        borderRight: "1px solid #ddd",
        padding: "1rem",
        background: "#1a1a1a",
        overflowY: "auto",
      }}
    >
      {/* ---------------------- 프로젝트 목록 ---------------------- */}
      <div style={{ marginBottom: "1.2rem", color: "white" }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <b style={{color: "white"}}>📁 프로젝트</b>
          <button onClick={() => router.push("/project-select")}>+</button>
        </div>

        {projects.map((p) => (
          <div
            key={p.id}
            onClick={() => {
              setProjectId(p.id);
              localStorage.setItem("project_id", String(p.id));
            }}
            style={{
              padding: "6px",
              marginTop: "6px",
              borderRadius: "6px",
              cursor: "pointer",
              background: projectId === p.id ? "#4a5cff" : "#eee",
              color: projectId === p.id ? "white" : "black",
            }}
          >
            {p.name}
          </div>
        ))}
      </div>

      {/* ---------------------- 채팅 목록 ---------------------- */}
      <div style={{ marginBottom: "1.2rem", color: "white" }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <b style={{color: "white"}}>💬 채팅방</b>
          <button onClick={() => setShowModal(true)}>+</button>
        </div>

        {chats.map((c) => (
          <div
            key={c.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              padding: "6px",
              marginTop: "6px",
              borderRadius: "6px",
              cursor: "pointer",
              background: chatId === c.id ? "#4a5cff" : "#eee",
              color: chatId === c.id ? "white" : "black",
            }}
          >
            <span
              onClick={async () => {
                setChatId(c.id);
                localStorage.setItem("chat_id", String(c.id));

                const res = await fetch(`${backend}/chats/${c.id}`);
                const data = await res.json();
                setChatHistory(data.history || []);
              }}
              style={{ flex: 1 }}
            >
              {c.title}
            </span>

            <button
              onClick={async (e) => {
                e.stopPropagation();

                const ok = confirm(`"${c.title}" 채팅을 삭제할까요?`);
                if (!ok) return;

                await fetch(`${backend}/chats/${c.id}`, { method: "DELETE" });

                const res = await fetch(`${backend}/projects/${projectId}/chats`);
                const data = await res.json();
                setChats(data.chats || []);
              }}
            >
              🗑
            </button>
          </div>
        ))}
      </div>

      {/* ---------------------- 프로젝트 파일 목록 ---------------------- */}
      <div>
        <b style={{color: "white"}}>📂 프로젝트 파일</b>
        {projectFiles.length === 0 && (
          <p style={{ fontSize: "12px", color: "#777" }}>업로드된 파일 없음</p>
        )}

        {projectFiles.map((f) => (
          <div
            key={f.id}
            style={{
              fontSize: "13px",
              padding: "4px",
              marginTop: "4px",
              borderRadius: "4px",
              background: "#eee",
              color: "black",
            }}
          >
            📄 {f.filename}
          </div>
        ))}
      </div>

      {showModal && (
        <CreateChatModal
          onClose={() => setShowModal(false)}
          onCreate={handleCreateChat}
          loraList={loraList}
        />
      )}
    </aside>
  );
}
