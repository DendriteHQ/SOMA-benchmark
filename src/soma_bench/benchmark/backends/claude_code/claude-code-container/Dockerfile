FROM node:22-bookworm-slim

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    bash \
    curl \
    ripgrep \
    python3 \
    python3-pip \
    openssh-client \
  && rm -rf /var/lib/apt/lists/*

# Both versions are pinned rather than @latest, because @latest is not reproducible here:
# Docker keys a RUN layer on its command string, and that string never changes. The image
# would keep whatever npm/pip resolved the first time it was built, and neither
# `docker build` nor `build --pull` would re-fetch it — so an image can sit many releases
# behind while the Dockerfile still appears to ask for the newest.
#
# Referencing the ARG inside the RUN makes a version change invalidate that layer by
# itself, so no --no-cache is required, and each bump is a reviewable one-line diff.

# The Agent SDK is a wrapper: it spawns this CLI as a subprocess and speaks its
# stream-json protocol, so the CLI has to be present even though nothing execs it directly.
ARG CLAUDE_CODE_VERSION=2.1.274
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

ARG CLAUDE_AGENT_SDK_VERSION=0.2.154
RUN pip3 install --no-cache-dir --break-system-packages \
      claude-agent-sdk==${CLAUDE_AGENT_SDK_VERSION}

# Recorded in the image so a consumer can assert which versions actually got installed
# instead of inferring them from the image tag.
LABEL org.opencontainers.image.version="${CLAUDE_CODE_VERSION}"
LABEL ai.soma.claude-agent-sdk.version="${CLAUDE_AGENT_SDK_VERSION}"

COPY agent_runner.py /usr/local/bin/agent_runner.py
RUN chmod 0555 /usr/local/bin/agent_runner.py

# Non-root, and not incidentally: `permission_mode="bypassPermissions"` is refused by the
# CLI when it runs as uid 0. The host seeds the workspace volume owned by this uid/gid —
# it reads them back out of the image with `docker inspect`, so they stay in one place.
RUN useradd -m -s /bin/bash claude
USER claude

# Claude Code writes its session state under $HOME/.claude; compose gives that a volume so
# it is scoped to the run rather than baked into the image layer.
ENV HOME=/home/claude

WORKDIR /testbed

ENTRYPOINT ["python3", "/usr/local/bin/agent_runner.py"]
