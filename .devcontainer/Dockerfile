FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install PyQt6
ENV MPLBACKEND=QtAgg

RUN python -m pip install --upgrade pip setuptools wheel

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

RUN python -m pip install maturin ruff ipykernel jupyter
# PyTorch CUDA install happens here.
ARG TORCH_VERSION=2.8.0+cu129
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN python -m pip install --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION}

WORKDIR /workspaces/mrtini