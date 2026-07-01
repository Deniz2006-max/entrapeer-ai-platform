import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Proxy /api/v1/* → backend so the browser never hits a cross-origin URL.
  // BACKEND_URL is a server-side env var (resolved at request time, not build time).
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://web:8000";
    return [
      {
        source: "/api/v1/:path*",
        destination: `${backendUrl}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
