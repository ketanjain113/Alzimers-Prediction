#!/bin/sh
set -e
# Repository-level start script. This is a safe fallback for hosting providers
# that deploy from the repo root and expect a start script (Railpack/Heroku-style).
# It will change into the Node server folder and start the Node app.

echo "Starting node_server from repo root..."
cd node_server

# Use npm ci if a lockfile exists, otherwise install
if [ -f package-lock.json ] || [ -f npm-shrinkwrap.json ]; then
  npm ci --production
else
  npm install --production
fi

npm start
