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

## Deploy via Docker on Railway (recommended single-container)

This repository includes a `Dockerfile` that creates a single container running both the
Flask model API and the Node server. Railway can build and run this image directly.

Quick steps:

1. In Railway, create a new project and choose "Deploy from GitHub" (connect your repo).
2. When Railway prompts for a Dockerfile, it will detect the root `Dockerfile` automatically.
3. Deploy — Railway will build the image. The container will start both services using
   `start.sh`.

Notes:
- When deploying a single Docker container on Railway, Railway will set a single `$PORT`
  that the platform expects your main web process to bind to. To avoid both services
  trying to bind the same port, set the following environment variable in Railway:
  - `MODEL_PORT=5000` (the model API will bind to this port)
  Railway will set `$PORT` (e.g., 3000) which the Node server will use. The Node server
  forwards requests to the model API at `http://127.0.0.1:5000` inside the container by default.
- If you prefer separate Railway services (recommended in some cases), deploy `Model` as
  one service (set Root Directory to `Model`) and `node_server` as another (Root Directory
  `node_server`). Then set `MODEL_API_URL` in the Node service to the Model service URL.

Troubleshooting:
- TensorFlow wheels are large — building the Docker image may take a while.
- If Railway's build times out or you hit resource limits, consider hosting the model in a
  separate service provider (Blob/S3 + download at startup) or using a smaller model.

