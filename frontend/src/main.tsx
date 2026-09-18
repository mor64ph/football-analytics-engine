import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Fire a wake request before React initializes. Render's free tier takes ~50s
// to cold-boot; this buys back the ~1s that would otherwise be lost to bundle
// parse + component mount before the first real API call.
const _apiBase = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
fetch(`${_apiBase}/health`, { keepalive: true }).catch(() => {});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
