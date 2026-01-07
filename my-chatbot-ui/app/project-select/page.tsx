"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function ProjectSelectPage() {
  const router = useRouter();
  const backend = "http://localhost:8000";

  const [user, setUser] = useState<any>(null);
  const [projects, setProjects] = useState<any[]>([]);

  // ✅ 프로젝트 생성용 입력값
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [persona, setPersona] = useState("");

  // ✅ 로그인 확인 + 프로젝트 로딩
  useEffect(() => {
    const u = localStorage.getItem("user");
    if (!u) {
      router.replace("/login");
      return;
    }

    const parsed = JSON.parse(u);
    setUser(parsed);

    fetch(`${backend}/projects?user_id=${parsed.id}`)
      .then((res) => res.json())
      .then((data) => {
        setProjects(data.projects || []);
      });
  }, []);

  // ✅ 기존 프로젝트 선택
  function handleSelectProject(projectId: number) {
    localStorage.setItem("project_id", String(projectId));
    router.replace("/");
  }

  // ✅ 새 프로젝트 생성
  async function handleCreateProject() {
    if (!name) {
        alert("프로젝트 이름은 필수입니다.");
        return;
    }

    if (!user?.id) {
        alert("로그인 정보가 없습니다. 다시 로그인하세요.");
        router.replace("/login");
        return;
    }

    const res = await fetch(`${backend}/projects`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        user_id: user.id,
        name,
        description,   // ✅ persona는 아직 안 보냄 (백엔드 구조 유지)
        }),
    });

    if (!res.ok) {
        const err = await res.text();
        console.error("❌ 프로젝트 생성 실패:", err);
        alert("프로젝트 생성 실패");
        return;
    }

    const data = await res.json();
    localStorage.setItem("project_id", String(data.project_id));
    router.replace("/");
    }


  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f6fb",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <div
        style={{
          width: "900px",
          background: "black",
          padding: "2.5rem",
          borderRadius: "16px",
          boxShadow: "0 10px 30px rgba(0,0,0,0.15)",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "2rem",
          color: "white",
        }}
      >
        {/* ✅ 왼쪽: 프로젝트 선택 */}
        <div>
          <h2 style={{ marginBottom: "1rem" }}>📁 내 프로젝트</h2>

          {projects.length === 0 && (
            <p style={{ color: "#888" }}>아직 생성된 프로젝트가 없습니다.</p>
          )}

          {projects.map((p) => (
            <div
              key={p.id}
              onClick={() => handleSelectProject(p.id)}
              style={{
                padding: "12px",
                background: "#f2f2f2",
                borderRadius: "10px",
                marginBottom: "10px",
                cursor: "pointer",
                color: "black",
              }}
            >
              <b>{p.name}</b>
              <p style={{ fontSize: "13px", color: "#666" }}>
                {p.description}
              </p>
            </div>
          ))}
        </div>

        {/* ✅ 오른쪽: 프로젝트 생성 */}
        <div>
          <h2 style={{ marginBottom: "1rem" }}>➕ 새 프로젝트 생성</h2>

          <input
            placeholder="프로젝트 이름"
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={inputStyle}
          />

          <textarea
            placeholder="프로젝트 설명 (예: 논문 요약, 계약서 분석 등)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            style={{ ...inputStyle, height: "80px" }}
          />

          <textarea
            placeholder="이 챗봇의 말투 / 성격 / 역할 (LoRA 기준)"
            value={persona}
            onChange={(e) => setPersona(e.target.value)}
            style={{ ...inputStyle, height: "100px" }}
          />

          <button
            onClick={handleCreateProject}
            style={{
              width: "100%",
              padding: "12px",
              marginTop: "10px",
              background: "#4a5cff",
              color: "white",
              border: "none",
              borderRadius: "10px",
              fontSize: "15px",
              cursor: "pointer",
            }}
          >
            프로젝트 생성하고 시작하기 🚀
          </button>
        </div>
      </div>
    </div>
  );
}

// ✅ 입력 공통 스타일
const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "10px",
  borderRadius: "8px",
  border: "1px solid #ccc",
  marginBottom: "10px",
};
