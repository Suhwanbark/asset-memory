# Ported from finautorubric resources/Dockerfile: same pinned base and Python lock,
# Codex CLI only (no Claude Code), Codex pinned to the version used on the Mac host.
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5
LABEL org.opencontainers.image.title="asset-memory-sandbox" \
      org.opencontainers.image.base.name="node:22-bookworm-slim" \
      org.opencontainers.image.base.digest="sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5" \
      io.linq.sandbox.environment-revision="asset-memory.sandbox.env/v1"
ARG CODEX_VERSION=0.159.2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv bash git ripgrep ca-certificates curl \
    jq ruby poppler-utils file binutils xxd unzip procps \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global @openai/codex@${CODEX_VERSION} \
    && npm cache clean --force \
    && python3 -m venv /opt/agent-venv
COPY requirements.lock environment_check.py /opt/sandbox-environment/
RUN /opt/agent-venv/bin/python -m pip install --require-hashes \
        --no-cache-dir -r /opt/sandbox-environment/requirements.lock \
    && mkdir -p /opt/sandbox-environment/manifest \
    && dpkg-query -W -f='${Package}\t${Version}\n' | sort \
        > /opt/sandbox-environment/manifest/os-packages.txt \
    && /opt/agent-venv/bin/python -m pip list --format=json \
        > /opt/sandbox-environment/manifest/python-packages.json \
    && chmod -R a-w /opt/agent-venv /opt/sandbox-environment \
    && printf '#!/bin/sh\nexec /opt/agent-venv/bin/python "$@"\n' > /usr/local/bin/python \
    && printf '#!/bin/sh\nexec /opt/agent-venv/bin/python3 "$@"\n' > /usr/local/bin/python3 \
    && chmod 0755 /usr/local/bin/python /usr/local/bin/python3 \
    && ln -sf /opt/agent-venv/bin/pip /usr/local/bin/pip \
    && ln -sf /opt/agent-venv/bin/pip3 /usr/local/bin/pip3 \
    && mkdir -p /work /home/agent \
    && chown 1000:1000 /work /home/agent
ENV HOME=/home/agent CODEX_HOME=/home/agent/.codex \
    PATH=/opt/agent-venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
USER 1000:1000
WORKDIR /work
CMD ["sleep", "infinity"]
