import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const pagesBase = process.env.VITE_BASE || (process.env.GITHUB_REPOSITORY ? `/${process.env.GITHUB_REPOSITORY.split("/")[1]}/` : "/");

export default defineConfig({
  plugins: [react()],
  base: pagesBase,
  server: {
    // Bind IPv4 loopback — Astrill/VPN often breaks localhost / ::1
    host: "127.0.0.1",
    port: 8080,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
