FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/mrtini-venv \
    MPLBACKEND=Agg \
    CARGO_TARGET_DIR=/opt/mrtini-cargo-target
ENV PATH="/opt/mrtini-venv/bin:/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates pkg-config \
    libssl-dev libffi-dev libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/mrtini-venv \
    && python -m pip install --upgrade pip setuptools wheel
RUN curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/mrtini-rustup.sh \
    && sh /tmp/mrtini-rustup.sh -y --profile minimal \
    && rm /tmp/mrtini-rustup.sh

ARG TORCH_VERSION=2.8.0+cu129
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN python -m pip install --index-url ${TORCH_INDEX_URL} "torch==${TORCH_VERSION}"

WORKDIR /workspaces/mrtini
# The mounted checkout is installed once by post_create.sh. No legacy engine
# build, patch installer, separate fast package, or GUI backend is needed.
