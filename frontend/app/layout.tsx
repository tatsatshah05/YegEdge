import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "YegEdge Terminal",
  description: "Bloomberg-style trading terminal",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
