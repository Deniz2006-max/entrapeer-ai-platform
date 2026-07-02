import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Server-side proxy fallback: if the browser hits a relative /api/* or /v1/*
  // URL, the Next.js standalone server forwards it to the backend container.
  // BACKEND_URL is set at runtime (docker-compose environment) — Next.js re-reads
  // it on each startup so the value is always current.
  //
  // Primary path: NEXT_PUBLIC_API_URL is baked at build time (Dockerfile ARG)
  // and points to http://localhost:8000 so client-side fetches go directly to
  // the host-exposed port — no proxy hop needed.
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://web:8000";
    return [
      // Legacy synchronous endpoints
      {
        source: "/api/v1/:path*",
        destination: `${backendUrl}/api/v1/:path*`,
      },
      // New Celery-backed unified endpoint (no /api prefix)
      {
        source: "/v1/:path*",
        destination: `${backendUrl}/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
