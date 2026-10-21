import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI backend so the console can be
// developed against a local API without CORS setup.
export default defineConfig({
  plugins: [react()],
  // The console is served by FastAPI at /console, so asset URLs must be
  // relative to that path rather than the server root.
  base: "/console/",
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.ARGUS_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: process.env.ARGUS_API_URL || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
