import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig(({ mode }) => {
  // loadEnv avoids depending on `process`, which would drag @types/node into the
  // browser type graph for the sake of one string.
  const env = loadEnv(mode, '.', '')
  const PROXY_TARGET = env.VOLCAST_PROXY_TARGET || 'http://127.0.0.1:8000'
  // Same-origin in production (the FastAPI app serves dist/). In dev the Vite
  // proxy forwards /api, so "" is right in both cases.
  const API_BASE = env.VITE_API_BASE ?? ''

  return {
    plugins: [
      react(),
      VitePWA({
        registerType: 'autoUpdate',
        includeAssets: ['favicon.svg'],
        manifest: {
          name: 'volcast',
          short_name: 'volcast',
          description: 'Short-horizon volatility forecasting and net-of-cost strategy research',
          theme_color: '#0b0f14',
          background_color: '#0b0f14',
          display: 'standalone',
          start_url: '/',
          icons: [
            { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
            { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
            {
              src: 'icon-512.png',
              sizes: '512x512',
              type: 'image/png',
              purpose: 'maskable',
            },
          ],
        },
        workbox: {
          // Never cache /api: stale forecasts are worse than a spinner.
          navigateFallbackDenylist: [/^\/api/],
          globPatterns: ['**/*.{js,css,html,svg,png,woff2}'],
        },
      }),
    ],
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': { target: PROXY_TARGET, changeOrigin: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      target: 'es2020',
    },
    define: {
      __API_BASE__: JSON.stringify(API_BASE),
    },
  }
})
