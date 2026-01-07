"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const backend = "http://localhost:8000";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  async function handleLogin() {
    const res = await fetch(`${backend}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();

    if (!data.success) {
      alert("로그인 실패");
      return;
    }

    // ✅ 핵심: 로그인 성공 저장
    localStorage.setItem("user", JSON.stringify(data.user));

    // ✅ 메인으로 이동
    router.replace("/");
  }

  async function handleRegister() {
    const res = await fetch(`${backend}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();

    if (!data.success) {
      alert("이미 존재하는 아이디");
      return;
    }

    alert("회원가입 성공!");
  }

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#f5f5f5",
        color: "white",
      }}
    >
      <div
        style={{
          width: "350px",
          background: "black",
          padding: "2rem",
          borderRadius: "12px",
          boxShadow: "0 5px 15px rgba(0,0,0,0.15)",
        }}
      >
        <h2 style={{ marginBottom: "1rem" }}>🔐 로그인</h2>

        <input
          placeholder="아이디"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          style={{ width: "100%", padding: "10px", marginBottom: "10px", border: "1px solid white", borderRadius: "8px",}}
        />

        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ width: "100%", padding: "10px", marginBottom: "14px", border: "1px solid white", borderRadius: "8px",}}
        />

        <button
          onClick={handleLogin}
          style={{
            width: "100%",
            padding: "10px",
            marginBottom: "8px",
            background: "#4a5cff",
            border: "none",
            color: "white",
            borderRadius: "8px",
          }}
        >
          로그인
        </button>

        <button
          onClick={handleRegister}
          style={{
            width: "100%",
            padding: "10px",
            background: "#999",
            border: "none",
            color: "white",
            borderRadius: "8px",
          }}
        >
          회원가입
        </button>
      </div>
    </div>
  );
}
