"use client";

import { useState, useEffect } from "react";

export default function ChatPage() {
  const backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState("");
  const [projectId, setProjectId] = useState<number | null>(null);
  const [chatId, setChatId] = useState<number | null>(null);

  // 프로젝트 ID 로드 (localStorage 또는 기본값)
  useEffect(() => {
    const storedProjectId = localStorage.getItem("project_id");
    if (storedProjectId) {
      setProjectId(Number(storedProjectId));
    } else {
      // 기본 프로젝트 ID 사용 (백엔드의 DEFAULT_PROJECT_ID)
      setProjectId(1);
    }
  }, []);

  const sendMessage = async () => {
    if (!input.trim() || !projectId) return;

    // 사용자가 입력한 메시지 추가
    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);

    // /chat_stream 엔드포인트로 요청
    const response = await fetch(`${backend}/chat_stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        project_id: projectId,
        chat_id: chatId, // null이면 백엔드에서 자동 생성
        query: input,
      }),
    });

    if (!response.body) {
      alert("스트리밍 응답 실패");
      return;
    }

    // 스트리밍 응답 처리
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let botText = "";
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      botText += chunk;

      // 실시간으로 메시지 업데이트
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: botText,
        };
        return updated;
      });
    }

    // 백엔드에서 생성된 chat_id 저장 (첫 메시지인 경우)
    if (!chatId) {
      // chat_id는 백엔드에서 자동 생성되므로, 필요시 별도 API로 조회해야 함
      // 여기서는 간단히 처리
    }

    setInput("");
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">RAG Chatbot</h1>

      <div className="border p-4 rounded-lg h-96 overflow-y-auto bg-gray-100">
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-3 ${msg.role === "user" ? "text-right" : "text-left"}`}>
            <span
              className={`inline-block px-3 py-2 rounded-lg ${
                msg.role === "user" ? "bg-blue-300" : "bg-gray-300"
              }`}
            >
              {msg.content}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-4 flex gap-2">
        <input
          className="flex-1 border px-3 py-2 rounded"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="질문을 입력하세요..."
        />
        <button
          onClick={sendMessage}
          className="px-4 py-2 bg-blue-500 text-white rounded"
        >
          전송
        </button>
      </div>
    </div>
  );
}
