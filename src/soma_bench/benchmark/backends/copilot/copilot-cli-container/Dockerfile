FROM node:22-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git \
    docker.io \
    ca-certificates \
    bash \
    curl \
    openssh-client \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g @github/copilot@latest

RUN useradd -m -s /bin/bash copilot
USER copilot

WORKDIR /workspace

ENTRYPOINT ["copilot"]