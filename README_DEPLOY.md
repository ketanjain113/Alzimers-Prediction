# Deployment notes — Railway (quick steps)

This file explains the minimal steps to deploy the project to Railway as two services (recommended):

1) Model (Flask) service
- In Railway, create a new Project → Deploy from GitHub → choose this repository.
- When asked for a subdirectory (Root Directory), enter: `Model`
- Railway will use `Model/requirements.txt` and the `Model/Procfile` (runs `gunicorn ModelAPI:app`).
- If the model binary (`Alzimer.keras`) is not in the repo, upload it to object storage and set the `MODEL_FILE` environment variable in Railway.

2) Node (frontend + proxy) service
- Create a second Railway Project → Deploy from GitHub → same repository.
- Set Root Directory to: `node_server`
- In the Node service environment variables add:
  - `MODEL_API_URL` = `https://<your-model-service>.up.railway.app`
- Railway will run `npm install` and `npm start` (see `node_server/package.json`).

Fallback: single-service root deploy
- If you must deploy from the repository root (single service), the repo includes `start.sh` which will start the Node server by changing into `node_server` and running `npm start`.

Ports
- Node server default port: 5001 (but Railway provides `$PORT` at runtime)
- Model server default port: 5000 (Procfile uses `$PORT`)

Troubleshooting
- If Railpack complains "No start command found", ensure the service's Root Directory is set to the correct subfolder (no leading slash) or use the `start.sh` fallback.
- If TensorFlow install fails on Railway, consider building a Docker image with TF preinstalled or host the model binary externally and download at startup.

Local testing
- Run the model locally: `cd Model && pip install -r requirements.txt && python ModelAPI.py`
- Run the node server locally: `cd node_server && npm install && $env:MODEL_API_URL="http://127.0.0.1:5000"; node server.js`
