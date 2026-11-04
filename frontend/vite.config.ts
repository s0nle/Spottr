import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The dashboard talks to spottr_app.py on ws://127.0.0.1:8765/ws.
// `npm run build` -> dist/, which spottr_app.py also serves at http://127.0.0.1:8765
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
})
