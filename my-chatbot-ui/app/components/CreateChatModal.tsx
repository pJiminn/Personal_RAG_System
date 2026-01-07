"use client";

import { useEffect, useState } from "react";
type CreateChatModalProps = {
  onClose: () => void;
  onCreate: (data: any) => void;
  loraList: any[];
};

export default function CreateChatModal({
  onClose,
  onCreate,
  loraList,
}: CreateChatModalProps) {

  const [mode, setMode] = useState("none"); // none | existing | new
  const [title, setTitle] = useState("");
  const [selectedLora, setSelectedLora] = useState(null);

  // 새 LoRA
  const [newName, setNewName] = useState("");
  const [newPurpose, setNewPurpose] = useState("");
  const [newModel, setNewModel] = useState("mistralai/Mistral-7B-Instruct-v0.2");

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.4)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 99999,
      }}
    >
      <div
        style={{
          width: "420px",
          background: "#1a1a1a",
          borderRadius: "12px",
          padding: "20px",
          color: "white",
        }}
      >
        <h2>🆕 새 채팅 만들기</h2>

        {/* 🔥 공통: 채팅 이름 */}
        <div style={{ marginTop: "12px" }}>
          <input
            placeholder="채팅 이름을 입력하세요"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            style={{
              width: "100%",
              padding: "8px",
              borderRadius: "6px",
              border: "1px solid #ddd",
              marginBottom: "12px",
            }}
          />
        </div>

        {/* ================================
              1) 일반 채팅
           ================================ */}
        <div>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "none"}
              onChange={() => setMode("none")}
            />
            일반 채팅 (LoRA 없음)
          </label>
        </div>

        {/* ================================
              2) 기존 LoRA 연결
           ================================ */}
        <div style={{ marginTop: "10px" }}>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "existing"}
              onChange={() => setMode("existing")}
            />
            기존 LoRA 선택
          </label>

          {mode === "existing" && (
            <div style={{ marginTop: "10px", marginLeft: "20px" }}>
              {loraList.length === 0 ? (
                <p style={{ color: "#777" }}>등록된 LoRA가 없습니다.</p>
              ) : (
                loraList.map((l) => (
                  <div key={l.id} style={{ marginBottom: "4px" }}>
                    <label>
                      <input
                        type="radio"
                        name="lora"
                        onChange={() => setSelectedLora(l.id)}
                      />
                      {l.name} ({l.status})
                    </label>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* ================================
              3) 새로운 LoRA 만들기
           ================================ */}
        <div style={{ marginTop: "10px" }}>
          <label>
            <input
              type="radio"
              name="mode"
              checked={mode === "new"}
              onChange={() => setMode("new")}
            />
            새로운 LoRA 만들기
          </label>

          {mode === "new" && (
            <div style={{ marginTop: "10px", marginLeft: "20px" }}>
              <input
                placeholder="LoRA 이름"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                style={{
                  width: "100%",
                  padding: "6px",
                  border: "1px solid #ccc",
                  borderRadius: "6px",
                }}
              />

              <input
                placeholder="학습 목표(예: 내 스타일로 답변)"
                value={newPurpose}
                onChange={(e) => setNewPurpose(e.target.value)}
                style={{
                  width: "100%",
                  marginTop: "8px",
                  padding: "6px",
                  border: "1px solid #ccc",
                  borderRadius: "6px",
                }}
              />

              <select
                value={newModel}
                onChange={(e) => setNewModel(e.target.value)}
                style={{
                  marginTop: "8px",
                  width: "100%",
                  padding: "6px",
                  borderRadius: "6px",
                }}
              >
                <option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral 7B</option>
                <option value="llama">LLaMA 계열</option>
              </select>
            </div>
          )}
        </div>

        {/* ================================
              취소 / 확인 버튼
           ================================ */}
        <div style={{ marginTop: "20px", display: "flex", gap: "8px" }}>
          <button
            onClick={onClose}
            style={{
              flex: 1,
              padding: "10px",
              borderRadius: "8px",
              background: "#ddd",
              color: "black",
            }}
          >
            취소
          </button>

          <button
            style={{
              flex: 1,
              padding: "10px",
              borderRadius: "8px",
              background: "#1166ff",
              color: "white",
            }}
            onClick={() => {
              if (!title.trim()) {
                alert("채팅 이름을 입력해주세요.");
                return;
              }

              if (mode === "none") {
                onCreate({ title, lora_id: null });
              } else if (mode === "existing") {
                if (!selectedLora) {
                  alert("LoRA를 선택하세요.");
                  return;
                }
                onCreate({ title, lora_id: selectedLora });
              } else if (mode === "new") {
                if (!newName.trim() || !newPurpose.trim()) {
                  alert("LoRA 이름과 목적을 입력해주세요.");
                  return;
                }
                onCreate({
                  title,
                  newLora: {
                    name: newName,
                    purpose: newPurpose,
                    base_model: newModel,
                  },
                });
              }

              onClose();
            }}
          >
            확인
          </button>
        </div>
      </div>
    </div>
  );
}
