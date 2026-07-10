FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG BUILD_ID=dev
ARG SOURCE_COMMIT=unknown
ARG ANYSPLAT_COMMIT=5f5e208a7dd57d52e43ea0d553a95eab526e8775
ARG DISCOVERSE_COMMIT=d67f47c084aba0e0cf422a8725235f8b9238655a
ARG ACT_COMMIT=f8ab39c8d829d85571ab6d0250412ba4d0698fa0
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
COPY src ./src
COPY scripts ./scripts
COPY vendor/gs_playground /opt/gs_playground
COPY docker/anysplat-runtime-requirements.txt /tmp/anysplat-runtime-requirements.txt
COPY docker/worker-constraints.txt /tmp/worker-constraints.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt . && \
    python -m pip install --no-cache-dir torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt -r /tmp/anysplat-runtime-requirements.txt && \
    python -m pip install --no-cache-dir xformers==0.0.24 --no-deps && \
    python -m pip install --no-cache-dir torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html && \
    python -m pip install --no-cache-dir 'https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl' && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt motrixsim-core==0.8.0

RUN git clone --filter=blob:none --no-checkout https://github.com/InternRobotics/AnySplat.git /opt/real2sim_paper_runtime/AnySplat && \
    git -C /opt/real2sim_paper_runtime/AnySplat checkout "${ANYSPLAT_COMMIT}" && \
    test "$(git -C /opt/real2sim_paper_runtime/AnySplat rev-parse HEAD)" = "${ANYSPLAT_COMMIT}" && \
    sed -i 's/VGGT.from_pretrained("facebook\/VGGT-1B")/VGGT.from_pretrained("facebook\/VGGT-1B", revision=os.environ.get("REAL2SIM_VGGT_MODEL_REVISION"))/g' /opt/real2sim_paper_runtime/AnySplat/src/model/encoder/anysplat.py

RUN git clone --filter=blob:none --no-checkout https://github.com/eust-w/DISCOVERSE.git /opt/DISCOVERSE && \
    git -C /opt/DISCOVERSE checkout "${DISCOVERSE_COMMIT}" && \
    git -C /opt/DISCOVERSE submodule update --init policies/act && \
    test "$(git -C /opt/DISCOVERSE rev-parse HEAD)" = "${DISCOVERSE_COMMIT}" && \
    test "$(git -C /opt/DISCOVERSE/policies/act rev-parse HEAD)" = "${ACT_COMMIT}" && \
    grep -Ev '^[[:space:]]*(torchvision|torch|opencv-python)([[:space:]]|$)' /opt/DISCOVERSE/policies/act/requirements/train_eval.txt > /tmp/discoverse-act-train-requirements.txt && \
    python -m pip install --no-cache-dir -c /tmp/worker-constraints.txt -r /tmp/discoverse-act-train-requirements.txt && \
    python /app/scripts/prepare_discoverse_act_runtime.py --act-root /opt/DISCOVERSE/policies/act --expected-commit "${ACT_COMMIT}" && \
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

RUN python -c 'import mujoco, motrixsim, torch; from huggingface_hub import snapshot_download; assert torch.__version__.startswith("2.2")' && \
    PYTHONPATH=/opt/DISCOVERSE/policies/act:/opt/DISCOVERSE python -c 'from policies.act.act import ACTPolicy; assert ACTPolicy is not None' && \
    PYTHONPATH=/opt/real2sim_paper_runtime/AnySplat python -c 'from src.model.model.anysplat import AnySplat; assert AnySplat is not None' && \
    test -f /opt/gs_playground/scripts/real2sim_video_motrix_scene.py

ENV REAL2SIM_PLATFORM_BUILD_ID=${BUILD_ID} \
    REAL2SIM_SOURCE_COMMIT=${SOURCE_COMMIT} \
    REAL2SIM_MOTRIXSIM_DOCS_COMMIT=${MOTRIXSIM_DOCS_COMMIT} \
    REAL2SIM_MOTRIXSIM_PYTHON=/opt/conda/bin/python \
    REAL2SIM_FFMPEG_BIN=/usr/bin/ffmpeg \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin real2sim && \
    chown -R 10001:10001 /app
USER 10001:10001
CMD ["real2sim-worker"]
