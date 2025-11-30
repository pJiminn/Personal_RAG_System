"use client";

import { useState } from "react";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    setLoading(true);
    setResults([]);

    const res = await fetch("http://127.0.0.1:8000/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await res.json();
    setResults(data.results || []);
    setLoading(false);
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">🔍 문서 검색</h1>

      <input
        type="text"
        placeholder="검색어를 입력하세요"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full border p-2 rounded mb-4"
      />

      <button
        onClick={handleSearch}
        className="bg-blue-500 text-white px-4 py-2 rounded"
      >
        검색
      </button>

      {loading && <p className="mt-4">검색 중...</p>}

      <div className="mt-6">
        {results.length > 0 &&
          results.map((item, index) => (
            <div key={index} className="border p-3 rounded mb-2">
              <p className="font-bold">문서 {index + 1}</p>
              <p className="text-gray-700">{item.text}</p>
              <p className="text-sm text-gray-400 mt-1">
                score: {item.score.toFixed(4)}
              </p>
            </div>
          ))}
      </div>

      {results.length === 0 && !loading && (
        <p className="mt-6 text-gray-400">검색 결과가 없습니다.</p>
      )}
    </div>
  );
}
