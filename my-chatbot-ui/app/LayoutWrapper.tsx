// LayoutWrapper.tsx
"use client";

export default function LayoutWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        width: "100vw",
        overflow: "hidden",
      }}
    >
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}
