import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

// Inside Docker Compose the frontend and backend are different containers,
// so "localhost" from inside the frontend container is itself, not the
// backend. VITE_BACKEND_URL (set in docker-compose.yml to the "backend"
// service name) overrides this; local `npm run dev` on the host keeps the
// localhost default since the backend is reachable there directly.
const backendTarget = process.env.VITE_BACKEND_URL || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(rootDir, "./src"),
    },
  },
  server: {
    host: true,
    port: 5173,
    // Docker Desktop on Windows doesn't forward real filesystem events for
    // bind-mounted volumes, so chokidar's native watcher silently misses
    // edits (same class of issue as the backend's watchfiles crash) and the
    // dev server keeps serving stale modules. Polling avoids relying on
    // those events.
    watch: {
      usePolling: true,
      interval: 300,
    },
    proxy: {
      // The WebSocket endpoint lives at /api/v1/ws/twin (see
      // backend/app/api/v1/ws.py), i.e. *under* /api — a separate "/ws"
      // proxy rule never matches it. `ws: true` on this single rule handles
      // both plain HTTP and the WebSocket upgrade for every /api/* path.
      "/api": { target: backendTarget, changeOrigin: true, ws: true },
    },
  },
});
