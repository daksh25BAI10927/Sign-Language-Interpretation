#!/usr/bin/env bash
# Build script for Render (Native Python runtime)
set -e

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Downloading Linux system libraries for MediaPipe ==="
mkdir -p ./libs
cd ./libs

# Try downloading libgles2 if apt-get is available
if command -v apt-get >/dev/null 2>&1; then
    apt-get download libgles2 libglvnd0 || true
    for f in *.deb; do
        if [ -f "$f" ]; then
            dpkg -x "$f" . && rm -f "$f"
        fi
    done
fi
cd ..

echo "=== Build finished successfully ==="
