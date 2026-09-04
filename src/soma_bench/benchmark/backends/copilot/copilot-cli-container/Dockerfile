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

# Pinned, not @latest. Docker caches a RUN layer on its command string, and that
# string never changes with @latest — so the image keeps whatever npm considered
# newest the first time anybody built it, and neither `build` nor `build --pull`
# re-fetches. In practice the team was running 1.0.71 (2026-07-16) while npm's latest
# was 1.0.82: eleven stable releases behind, without knowing it.
#
# That matters beyond hygiene. The Copilot CLI has renamed the assistant reasoning
# field three times (`reasoning` -> `reasoning_text` in 1.0.81 -> `reasoning_content`),
# and the compressor keys off those names. 1.0.71 sends the original one, so a stale
# image silently benchmarks a wire format no current user sends — and passes.
#
# Referencing the ARG inside the RUN means a bump invalidates this layer by itself, so
# no --no-cache is needed, and the version is a reviewable one-line diff.
ARG COPILOT_VERSION=1.0.82
RUN npm install -g @github/copilot@${COPILOT_VERSION}

# Recorded in the image so a consumer can assert what actually got installed rather
# than trusting the tag: SOMA-run-local's tests/test_copilot_version.py compares this
# pin, the version inside the built image, and npm's dist-tag.
LABEL org.opencontainers.image.version="${COPILOT_VERSION}"

RUN useradd -m -s /bin/bash copilot
USER copilot

WORKDIR /workspace

ENTRYPOINT ["copilot"]