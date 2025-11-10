#!/bin/sh
set -e
# Repository-level start script used by the Dockerfile / Railway container.
# It starts the Flask model API in the background and then starts the Node server.

echo "Starting Model API (background)..."
cd /app/Model || cd Model

# Ensure MODEL_PORT is set (default 5000) and MODEL_API_URL points to it
export MODEL_PORT=${MODEL_PORT:-5000}
export MODEL_API_URL=${MODEL_API_URL:-http://127.0.0.1:${MODEL_PORT}}

# Make logs directory
mkdir -p /app/logs
echo "Model logs -> /app/logs/model.log"

# Start Python model in background and redirect output to log
python ModelAPI.py > /app/logs/model.log 2>&1 &
MODEL_PID=$!

# Wait for the model /health to return 200 (timeout after 60s)
echo "Waiting for model to become ready on ${MODEL_API_URL}/health..."
READY=0
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
	# Use curl to check health; suppress output
	HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${MODEL_API_URL}/health" || true)
	if [ "$HTTP_CODE" = "200" ]; then
		READY=1
		break
	fi
	sleep 1
	WAITED=$((WAITED + 1))
done

if [ $READY -ne 1 ]; then
	echo "Model did not become ready after ${MAX_WAIT}s. Dumping model log:" >&2
	cat /app/logs/model.log >&2 || true
	echo "Killing model process (pid=${MODEL_PID})" >&2 || true
	kill ${MODEL_PID} 2>/dev/null || true
	exit 1
fi

echo "Model is ready — starting Node server (foreground)..."
cd /app/node_server || cd node_server
# Start Node server in foreground so container keeps running
npm start
