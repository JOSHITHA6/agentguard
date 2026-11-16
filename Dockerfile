FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# Offline demo by default; for a real model point --host at your Ollama server,
# e.g. docker run --add-host=host.docker.internal:host-gateway agentguard \
#   python -m agentguard "your question" --host http://host.docker.internal:11434
CMD ["python", "-m", "agentguard", "--demo"]
