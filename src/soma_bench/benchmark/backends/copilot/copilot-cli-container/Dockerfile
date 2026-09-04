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

# Pinned rather than @latest, because @latest is not reproducible here: Docker keys a
# RUN layer on its command string, and that string never changes. The image therefore
# keeps whatever npm resolved the first time it was built, and neither `docker build`
# nor `build --pull` will re-fetch it — so an image can sit many releases behind while
# the Dockerfile still appears to ask for the newest.
#
# Referencing the ARG inside the RUN makes a version change invalidate this layer by
# itself, so no --no-cache is required, and the version is a reviewable one-line diff.
ARG COPILOT_VERSION=1.0.82
RUN npm install -g @github/copilot@${COPILOT_VERSION}

# Recorded in the image so a consumer can assert which version actually got installed
# instead of inferring it from the image tag.
LABEL org.opencontainers.image.version="${COPILOT_VERSION}"

RUN useradd -m -s /bin/bash copilot
USER copilot

WORKDIR /workspace

ENTRYPOINT ["copilot"]