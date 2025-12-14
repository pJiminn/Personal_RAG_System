"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import Sidebar from "./sidebar";
import LayoutWrapper from "./LayoutWrapper";
import styles from "./ui.module.css";

export default function Home() {
  const router = useRouter();
  const backend = "http://localhost:8000";

  // ======================================================
  // 1) 모든 상태는 최상단에 선언해야 함 !!
  // ======================================================
  const [authChecked, setAuthChecked] = useState(false);
  const [user, setUser] = useState<any>(null);

  const [projectId, setProjectId] = useState<number | null>(null);
  const [chatList, setChatList] = useState<any[]>([]);
  const [chatId, setChatId] = useState<number | null>(null);

  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [chatInput, setChatInput] = useState("");

  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState("");

  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);

  const [chatFiles, setChatFiles] = useState<any[]>([]);

  const [showUserSwitcher, setShowUserSwitcher] = useState(false);
  const [userList, setUserList] = useState<any[]>([]);

  const [showLoraModal, setShowLoraModal] = useState(false);
  const [loraName, setLoraName] = useState("");
  const [loraDesc, setLoraDesc] = useState("");
  const [currentLoraId, setCurrentLoraId] = useState<number | null>(null);
  const [loraStatus, setLoraStatus] = useState<any | null>(null);

  // ======================================================
  // 2) 로그인 체크
  // ======================================================
useEffect(() => {
  const u = localStorage.getItem("user");

  if (!u) {
    router.replace("/login");
    return;
  }

  try {
    const parsed = JSON.parse(u);

    // 여기서 반드시 유저 id 존재 여부 확인
    if (!parsed?.id) {
      router.replace("/login");
      return;
    }

    setUser(parsed);

    // 🔥 user 정보가 완전하게 설정된 뒤에서야 authChecked 를 true 로 변경
    setAuthChecked(true);

  } catch (err) {
    // JSON 파싱 실패하면 로그인으로
    router.replace("/login");
  }
}, []);




// ======================================================
// 3) 로그인 이후 프로젝트 리스트 로딩
// ======================================================
useEffect(() => {
  if (!user) return;

  async function loadProjects() {
    const res = await fetch(`${backend}/projects?user_id=${user.id}`);
    const data = await res.json();

    if (!data.projects || data.projects.length === 0) {
      router.replace("/project-select");
      return;
    }

    const defaultProject = data.projects[0].id;
    setProjectId(defaultProject);
    localStorage.setItem("project_id", String(defaultProject));
  }

  loadProjects();
}, [user]);



  // ======================================================
  // ✅ 2. 채팅방 목록 불러오기 (project 기준)
  // ======================================================
  useEffect(() => {
    if (!projectId) return;

    async function loadChats() {
      const res = await fetch(`${backend}/projects/${projectId}/chats`);
      const data = await res.json();

      setChatList(data.chats || []);

      if (data.chats?.length > 0) {
        setChatId(data.chats[0].id);
        localStorage.setItem("chat_id", String(data.chats[0].id));
      } else {
        setChatId(null);
        setChatHistory([]);
      }
    }

    loadChats();
  }, [projectId]);

  // ======================================================
  // ✅ 3. 채팅 히스토리 로딩 (chat 기준)
  // ======================================================
  useEffect(() => {
    if (!chatId) return;

    async function loadHistory() {
      const res = await fetch(`${backend}/chats/${chatId}`);
      const data = await res.json();
      setChatHistory(data.history || []);
    }

    loadHistory();
  }, [chatId]);

  // ======================================================
  // ✅ 4. 새 채팅 생성 (이름 입력형)
  // ======================================================
  async function createNewChat() {
    if (!projectId) return;

    const title = prompt("새 채팅 이름을 입력하세요");
    if (!title) return;

    const res = await fetch(`${backend}/projects/${projectId}/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });

    if (!res.ok) {
      alert("채팅 생성 실패");
      return;
    }

    const data = await res.json();
    setChatId(data.chat_id);
    localStorage.setItem("chat_id", String(data.chat_id));

    // ✅ 채팅 목록 갱신
    const chatRes = await fetch(`${backend}/projects/${projectId}/chats`);
    const chatData = await chatRes.json();
    setChatList(chatData.chats || []);

    setChatHistory([]);
  }


    // ✅ LoRA 학습 상태 폴링
  useEffect(() => {
    if (!currentLoraId) return;

    const interval = setInterval(async () => {
      const res = await fetch(`${backend}/lora/${currentLoraId}/status`);
      const data = await res.json();
      setLoraStatus(data);

      // 완료/에러 시 폴링 정지
      if (data.state === "done" || data.state === "error") {
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentLoraId]);

  // ======================================================
  // ✅ 5. 채팅 전송 (스트리밍)
  // ======================================================
  async function handleChat() {
    if (!chatInput.trim() || !projectId || !chatId) return;

    const userMsg = { role: "user", content: chatInput };
    setChatHistory((prev) => [...prev, userMsg]);

    const res = await fetch(`${backend}/chat_stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        chat_id: chatId,
        query: chatInput,
      }),
    });

    if (!res.body) {
      alert("스트리밍 응답 실패");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let botText = "";
    setChatHistory((prev) => [...prev, { role: "assistant", content: "" }]);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      botText += chunk;

      setChatHistory((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: botText,
        };
        return updated;
      });
    }

    setChatInput("");
  }



    // ✅ LoRA 생성 + 학습 시작
  async function handleCreateLora() {
    if (!projectId) {
      alert("먼저 프로젝트를 선택하세요.");
      return;
    }
    if (!loraName.trim()) {
      alert("LoRA 이름을 입력하세요.");
      return;
    }

    // 1) LoRA 프로필 생성
    const res = await fetch(`${backend}/lora`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        name: loraName,
        description: loraDesc,
        base_model: "mistralai/Mistral-7B-Instruct-v0.2",
      }),
    });

    const data = await res.json();
    const loraId = data.lora_id;
    setCurrentLoraId(loraId);

    // 2) 학습 시작
    await fetch(`${backend}/lora/${loraId}/train`, {
      method: "POST",
    });

    // 모달 닫기
    setShowLoraModal(false);
    setLoraName("");
    setLoraDesc("");
  }

  // ======================================================
  // ✅ 6. 파일 업로드
  // ======================================================
  async function handleUpload(e: any) {
    if (!projectId) return;

    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${backend}/projects/${projectId}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    alert(data.message);
    setUploading(false);
  }

  // ======================================================
  // ✅ 7. 문서 검색
  // ======================================================
  async function handleSearch() {
    if (!projectId) return;

    const res = await fetch(`${backend}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        query: searchText,
      }),
    });

    const data = await res.json();
    setSearchResults(data.results || []);
  }


  //로그아웃하수
  function handleLogout() {
  localStorage.removeItem("user");
  localStorage.removeItem("project_id");
  localStorage.removeItem("chat_id");
  router.replace("/login");
}

// 계정 전환
async function openUserSwitcher() {
  const res = await fetch(`${backend}/admin/users`);
  const data = await res.json();
  setUserList(data.users || []);
  setShowUserSwitcher(true);
}

  // ======================================================
  // ✅ ✅ ✅ 최종 렌더링
  // ======================================================
return (
  <LayoutWrapper>
    
    {!authChecked ? (
      <div
        style={{
          display: "flex",
          height: "100vh",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "18px",
        }}
      >

      </div>
    ) : (
    
    <div style={{ display: "flex", height: "100vh", width: "100vw" }}>
      {/* ✅ 사이드바 영역 (고정 폭) */}
      <div style={{ width: "260px", flexShrink: 0 }}>
        <Sidebar
          projectId={projectId}
          setProjectId={setProjectId}
          chatId={chatId}
          setChatId={setChatId}
          setChatHistory={setChatHistory}
        />
      </div>

      {/* ✅ 메인 컨텐츠 영역 */}
      <main
        style={{
          flex: 1,
          padding: "2rem",
          background: "#fafafa",
          overflowY: "auto",
        }}
      >
        {/* ✅ 로그인 계정 표시 바 (맨 위에) */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "10px 16px",
            background: "#111",
            color: "white",
            borderRadius: "8px",
            marginBottom: "16px",
          }}
        >
          <div>
            👤 로그인 계정:{" "}
            <b>{user?.email || user?.username || "알 수 없음"}</b>
          </div>

          <div style={{ display: "flex", gap: "8px" }}>
            <button onClick={handleLogout}>로그아웃</button>
            <button onClick={openUserSwitcher}>계정 전환</button>
          </div>
        </div>

        {/* ✅ 제목 */}
        <h1 className={styles.title}>📚 Personal RAG Chatbot</h1>

        {/* ✅ 업로드 / 검색 */}
        <div className={styles.row}>
          <div className={styles.card}>
            <h3>📄 문서 업로드</h3>
            <input type="file" accept=".pdf" onChange={handleUpload} />
            <p>{fileName || "선택된 파일 없음"}</p>
            {uploading && <p>업로드 중...</p>}
          </div>

          <div className={styles.card}>
            <h3>🔍 문서 검색</h3>
            <input
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
            />
            <button onClick={handleSearch}>검색</button>

            {searchResults.map((r, i) => (
              <div key={i}>
                <b>Score:</b> {r.score}
                <p>{r.text}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ✅ 채팅 */}
        <div className={styles.chatBox}>
          <div className={styles.chatWindow}>
            {chatHistory.map((msg, i) => (
              <div
                key={i}
                className={
                  msg.role === "user"
                    ? styles.userBubble
                    : styles.botBubble
                }
              >
                {msg.content}
              </div>
            ))}
          </div>

          <div className={styles.chatInputRow}>
            <input
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              placeholder="질문 입력"
            />
            <button onClick={handleChat}>전송</button>
          </div>
        </div>
      </main>
    </div>
    )}
    {showUserSwitcher && (
      <div
        style={{
          position: "fixed",
          inset: 0,
          background: "rgba(0,0,0,0.5)",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          zIndex: 9999,
        }}
      >
        <div
          style={{
            background: "white",
            padding: "20px",
            borderRadius: "12px",
            width: "320px",
          }}
        >
          <h3>👥 계정 전환</h3>

          {userList.map((u) => (
            <div
              key={u.id}
              onClick={() => {
                localStorage.setItem("user", JSON.stringify(u));
                localStorage.removeItem("project_id");
                localStorage.removeItem("chat_id");
                location.reload();
              }}
              style={{
                padding: "8px",
                cursor: "pointer",
                borderBottom: "1px solid #ddd",
              }}
            >
              {u.email || u.username}
            </div>
          ))}

          <button onClick={() => setShowUserSwitcher(false)}>닫기</button>
        </div>
      </div>
    )}
  </LayoutWrapper>
);
}