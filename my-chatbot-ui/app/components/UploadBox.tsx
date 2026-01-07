"use client";

import { useState } from "react";

export default function UploadBox() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) {
      setStatus("❌ 파일을 선택해주세요");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setStatus("업로드 중...");

    try {
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setStatus(`✅ 업로드 성공: ${data.message}`);
      } else {
        setStatus(`❌ 업로드 실패: ${data.detail}`);
      }
    } catch (error) {
      console.error(error);
      setStatus("❌ 서버 오류 발생");
    }
  };

  return (
    <div className="w-full max-w-md p-6 bg-white border rounded-xl shadow-md">
      <h2 className="text-xl font-semibold mb-3">📄 문서 업로드</h2>
      <input
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
        className="mb-3"
      />
      <button
        onClick={handleUpload}
        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition"
      >
        업로드
      </button>
      {status && <p className="mt-3 text-gray-700">{status}</p>}
    </div>
  );
}
