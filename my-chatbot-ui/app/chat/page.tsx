"use client";

import { useState } from "react";

export default function ChatPage() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input) return;

    // 사용자가 입력한 메시지 추가
    setMessages((prev) => [...prev, { role: "user", content: input }]);

    const response = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: input }),
    });

    const data = await response.json();

    // 백엔드 답변 추가
    setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);

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
