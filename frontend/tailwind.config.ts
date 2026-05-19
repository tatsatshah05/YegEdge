import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "Consolas", "monospace"],
      },
      colors: {
        terminal: {
          bg: "#0a0a0a",
          panel: "#0d1117",
          header: "#161b22",
          border: "#30363d",
          text: "#e6edf3",
          muted: "#8b949e",
          accent: "#f0883e",
          green: "#3fb950",
          red: "#f85149",
          blue: "#58a6ff",
          yellow: "#d29922",
        },
      },
    },
  },
  plugins: [],
};

export default config;
