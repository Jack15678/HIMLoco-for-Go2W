FROM public.ecr.aws/docker/library/python:3.8-slim-bookworm

RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    g++ ninja-build libgl1 libglib2.0-0 libxext6 libxrender1 libsm6 libxi6 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pip==25.0.1
RUN pip install --no-cache-dir --no-deps torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --no-deps triton==2.0.0 --index-url https://download.pytorch.org/whl/cu118
RUN --mount=type=cache,target=/root/.cache/pip pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    numpy==1.23.5 scipy==1.10.1 matplotlib==3.7.5 \
    tensorboard==2.14.0 ninja==1.11.1.4 pyyaml==6.0.2 mujoco==3.2.3 imageio==2.35.1 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple

ENV LD_LIBRARY_PATH=/usr/local/lib
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/him
