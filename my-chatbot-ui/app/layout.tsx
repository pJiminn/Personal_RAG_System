import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import LayoutWrapper from "./LayoutWrapper";   // ⬅ 반드시 추가

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RAG Chatbot",
  description: "RAG chatbot with sidebar & streaming",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {/* ▼▼▼ children을 LayoutWrapper로 감싸는 것이 중요 ▼▼▼ */}
        <LayoutWrapper>
          {children}
        </LayoutWrapper>
        {/* ▲▲▲ 여기만 바뀐 부분 ▲▲▲ */}
      </body>
    </html>
  );
}
