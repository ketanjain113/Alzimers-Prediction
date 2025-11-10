#!/bin/sh
set -e
# Repository-level start script used by the Dockerfile / Railway container.
# It starts the Flask model API in the background and then starts the Node server.

echo "Starting Model API (background)..."
cd /app/Model || cd Model
# Start Python model in background
python ModelAPI.py &

echo "Starting Node server (foreground)..."
cd /app/node_server || cd node_server
# Start Node server in foreground so container keeps running
npm start
