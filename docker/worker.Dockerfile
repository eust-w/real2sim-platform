FROM node:20.19.2-bookworm-slim@sha256:7cd3fbc830c75c92256fe1122002add9a1c025831af8770cd0bf8e45688ef661 AS node-runtime

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG ANYSPLAT_COMMIT=5f5e208a7dd57d52e43ea0d553a95eab526e8775
ARG DISCOVERSE_COMMIT=d67f47c084aba0e0cf422a8725235f8b9238655a
ARG ACT_COMMIT=f8ab39c8d829d85571ab6d0250412ba4d0698fa0
ARG DISCOVERSE_ARCHIVE_SHA256=c8fbd6ca571ce3d44a61c6d3903de956075ee28d29afdfbcfe2b3fcc4cdab350
ARG ACT_ARCHIVE_SHA256=35c5dfc416ba0a5d7f10ad61252fa9f685ac4cf2b123cd6f1b3db1619a2b1b37
ARG TORCH_SCATTER_SHA256=caab7dd4a8783d49b2a983331e93ebf968c9c76ff4daaca158e2f6fb9c8028bf
ARG MOTRIXSIM_DOCS_COMMIT=2beecea67c81d25e29cefa30a80c0ddad7b44dde
ARG VGGT_MODEL_REVISION=860abec7937da0a4c03c41d3c269c366e82abdf9

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REAL2SIM_VGGT_MODEL_REVISION=${VGGT_MODEL_REVISION} \
    REAL2SIM_ANYSPLAT_ROOT=/opt/real2sim_paper_runtime/AnySplat \
    REAL2SIM_ANY_SPLAT_PYTHON=/opt/real2sim_paper_runtime/envs/anysplat/bin/python \
    REAL2SIM_POSE_INIT_TRAIN_PYTHON=/opt/real2sim_paper_runtime/envs/anysplat/bin/python \
    CUDA_HOME=/usr/local/cuda \
    PYTHONPATH=/opt/real2sim_paper_runtime/AnySplat:/opt/DISCOVERSE/policies/act:/opt/DISCOVERSE:/opt/gs_playground

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      ca-certificates curl ffmpeg git git-lfs build-essential cmake ninja-build pkg-config colmap \
      libgl1 libglib2.0-0 libgomp1 libx11-6 libxext6 libsm6 && \
    git lfs install --system && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY docker/anysplat-runtime-requirements.txt /tmp/anysplat-runtime-requirements.txt
COPY docker/worker-constraints.txt /tmp/worker-constraints.txt
COPY scripts/prepare_discoverse_act_runtime.py /tmp/prepare_discoverse_act_runtime.py
COPY src/real2sim_platform/__init__.py /tmp/real2sim_platform/__init__.py
COPY src/real2sim_platform/source_revision.py /tmp/real2sim_platform/source_revision.py

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir tomli==2.2.1 && \
    python -c 'import tomli; config=tomli.load(open("pyproject.toml", "rb")); open("/tmp/platform-requirements.txt", "w").write("\n".join(config["project"]["dependencies"]) + "\n")' && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt -r /tmp/platform-requirements.txt && \
    python -m pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt -r /tmp/anysplat-runtime-requirements.txt && \
    python -m pip install --no-cache-dir xformers==0.0.24 --no-deps && \
    python -m pip install --no-cache-dir 'https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl' && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt motrixsim-core==0.8.0

RUN curl -fL --retry 8 --retry-all-errors --retry-delay 2 --connect-timeout 20 --max-time 300 --speed-limit 1024 --speed-time 30 \
      -o '/tmp/torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl' 'https://data.pyg.org/whl/torch-2.2.0%2Bcu121/torch_scatter-2.1.2%2Bpt22cu121-cp310-cp310-linux_x86_64.whl' && \
    echo "${TORCH_SCATTER_SHA256}  /tmp/torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl" | sha256sum -c - && \
    python -m pip install --no-cache-dir --no-deps '/tmp/torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl' && \
    rm -f '/tmp/torch_scatter-2.1.2+pt22cu121-cp310-cp310-linux_x86_64.whl'

RUN git clone --filter=blob:none --no-checkout https://github.com/InternRobotics/AnySplat.git /opt/real2sim_paper_runtime/AnySplat && \
    git -C /opt/real2sim_paper_runtime/AnySplat checkout "${ANYSPLAT_COMMIT}" && \
    test "$(git -C /opt/real2sim_paper_runtime/AnySplat rev-parse HEAD)" = "${ANYSPLAT_COMMIT}" && \
    sed -i 's/VGGT.from_pretrained("facebook\/VGGT-1B")/VGGT.from_pretrained("facebook\/VGGT-1B", revision=os.environ.get("REAL2SIM_VGGT_MODEL_REVISION"))/g' /opt/real2sim_paper_runtime/AnySplat/src/model/encoder/anysplat.py

RUN mkdir -p /opt/DISCOVERSE /opt/DISCOVERSE/policies/act && \
    curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 900 \
      -o /tmp/discoverse.tar.gz "https://codeload.github.com/eust-w/DISCOVERSE/tar.gz/${DISCOVERSE_COMMIT}" && \
    echo "${DISCOVERSE_ARCHIVE_SHA256}  /tmp/discoverse.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/discoverse.tar.gz --strip-components=1 -C /opt/DISCOVERSE && \
    rm -rf /opt/DISCOVERSE/policies/act && mkdir -p /opt/DISCOVERSE/policies/act && \
    curl -fL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 300 \
      -o /tmp/act.tar.gz "https://codeload.github.com/OpenGHz/Imitate-All/tar.gz/${ACT_COMMIT}" && \
    echo "${ACT_ARCHIVE_SHA256}  /tmp/act.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/act.tar.gz --strip-components=1 -C /opt/DISCOVERSE/policies/act && \
    printf '{"schema":"real2sim.source.v1","source":"https://codeload.github.com/eust-w/DISCOVERSE/tar.gz/%s","commit":"%s","archive_sha256":"%s"}\n' \
      "${DISCOVERSE_COMMIT}" "${DISCOVERSE_COMMIT}" "${DISCOVERSE_ARCHIVE_SHA256}" > /opt/DISCOVERSE/.real2sim-source.json && \
    printf '{"schema":"real2sim.source.v1","source":"https://codeload.github.com/OpenGHz/Imitate-All/tar.gz/%s","commit":"%s","archive_sha256":"%s"}\n' \
      "${ACT_COMMIT}" "${ACT_COMMIT}" "${ACT_ARCHIVE_SHA256}" > /opt/DISCOVERSE/policies/act/.real2sim-source.json && \
    rm -f /tmp/discoverse.tar.gz /tmp/act.tar.gz && \
    grep -Ev '^[[:space:]]*(torchvision|torch|opencv-python)([[:space:]]|$)' /opt/DISCOVERSE/policies/act/requirements/train_eval.txt > /tmp/discoverse-act-train-requirements.txt && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt -r /tmp/discoverse-act-train-requirements.txt && \
    PYTHONPATH=/tmp python /tmp/prepare_discoverse_act_runtime.py --act-root /opt/DISCOVERSE/policies/act --expected-commit "${ACT_COMMIT}" && \
    python -m pip install --no-cache-dir --no-deps -e /opt/DISCOVERSE

RUN git clone --filter=blob:none --no-checkout https://github.com/Motphys/motrixsim-docs.git /opt/motrixsim-docs && \
    git -C /opt/motrixsim-docs checkout "${MOTRIXSIM_DOCS_COMMIT}" && \
    git -C /opt/motrixsim-docs lfs pull --include='examples/assets/**' && \
    test "$(git -C /opt/motrixsim-docs rev-parse HEAD)" = "${MOTRIXSIM_DOCS_COMMIT}" && \
    test "$(stat -c%s /opt/motrixsim-docs/examples/assets/rm65_open_cabinet/assets/rm65/meshes/base_link.STL)" -gt 10000 && \
    ! grep -q 'git-lfs.github.com/spec' /opt/motrixsim-docs/examples/assets/rm65_open_cabinet/assets/rm65/meshes/base_link.STL && \
    git config --system --add safe.directory /opt/motrixsim-docs && \
    mkdir -p /opt/real2sim_paper_runtime/envs/anysplat/bin && \
    ln -s /opt/conda/bin/python /opt/real2sim_paper_runtime/envs/anysplat/bin/python && \
    ln -s /opt/conda/bin/pip /opt/real2sim_paper_runtime/envs/anysplat/bin/pip

# Keep the stable CUDA/ML dependency layers reusable when only the MotrixSim
# graphics runtime changes.
RUN apt-get -o Acquire::Retries=5 update && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends libvulkan1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node
RUN test "$(node --version)" = "v20.19.2"

COPY src ./src
COPY scripts ./scripts
COPY vendor/gs_playground /opt/gs_playground
RUN python -m pip install --no-cache-dir --no-deps .

RUN python -c 'import mujoco, motrixsim, torch, trimesh; from pxr import Usd, UsdGeom; from huggingface_hub import snapshot_download; assert torch.__version__.startswith("2.2"); assert Usd is not None and UsdGeom is not None and trimesh is not None' && \
    test "$(node --version)" = "v20.19.2" && \
    PYTHONPATH=/opt/DISCOVERSE/policies/act:/opt/DISCOVERSE python -c 'from policies.act.act import ACTPolicy; assert ACTPolicy is not None' && \
    PYTHONPATH=/opt/real2sim_paper_runtime/AnySplat python -c 'from src.model.model.anysplat import AnySplat; assert AnySplat is not None' && \
    test -f /opt/gs_playground/scripts/real2sim_video_motrix_scene.py

ARG BUILD_ID=dev
ARG SOURCE_COMMIT=unknown
ENV REAL2SIM_PLATFORM_BUILD_ID=${BUILD_ID} \
    REAL2SIM_SOURCE_COMMIT=${SOURCE_COMMIT} \
    REAL2SIM_MOTRIXSIM_DOCS_COMMIT=${MOTRIXSIM_DOCS_COMMIT} \
    REAL2SIM_MOTRIXSIM_PYTHON=/opt/conda/bin/python \
    REAL2SIM_FFMPEG_BIN=/usr/bin/ffmpeg \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin real2sim && \
    chown -R 10001:10001 /app
USER 10001:10001
CMD ["real2sim-worker"]
