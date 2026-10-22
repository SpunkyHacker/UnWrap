#!/bin/bash
# Expose Unwrap PWA over HTTPS via Cloudflare Quick Tunnel for phone camera testing.
set -e

if ! command -v cloudflared &> /dev/null; then
    echo "cloudflared not found. Installing via Homebrew..."
    brew install cloudflared
fi

echo "=========================================================="
echo "Starting Cloudflare HTTPS tunnel for Unwrap (Vite port 5173)"
echo "API calls to /api are automatically proxied to port 8000."
echo "=========================================================="
echo "Open the generated https://*.trycloudflare.com link on your mobile phone."
echo ""

cloudflared tunnel --url http://127.0.0.1:5173
