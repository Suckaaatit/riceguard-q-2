import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [react(), basicSsl()],
  base: "./",
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    https: true,
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      '/analyze': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
      ,
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/history': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/live_preview': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      }
    }
  }
  
})
        // target: 'http://localhost:8000',