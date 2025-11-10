FROM tensorflow/tensorflow:2.10.0

# Install Node.js 18 (required for the node_server)
RUN apt-get update && apt-get install -y curl ca-certificates gnupg2 && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy repo
COPY . /app

# Upgrade pip and install Python deps for the model
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r Model/requirements.txt || true

# Install Node dependencies for the Node server
WORKDIR /app/node_server
RUN npm install --production || true

# Expose common ports (Railway provides $PORT at runtime)
EXPOSE 5000 5001

WORKDIR /app

# Make sure start script is executable
RUN chmod +x /app/start.sh

# Default command: use repository start script which launches both services
CMD ["/app/start.sh"]
