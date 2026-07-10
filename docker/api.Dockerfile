FROM python:3.12-slim-bookworm

ARG BUILD_ID=dev
ARG SOURCE_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REAL2SIM_PLATFORM_BUILD_ID=${BUILD_ID} \
    REAL2SIM_SOURCE_COMMIT=${SOURCE_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir .
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin real2sim
USER 10001:10001
EXPOSE 18126
CMD ["real2sim-platform", "--host", "0.0.0.0", "--port", "18126"]
