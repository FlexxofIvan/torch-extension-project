ARG COURSE_IMAGE=ubuntu:24.04
FROM ${COURSE_IMAGE} AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    g++ \
    python3 \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \
    "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu \
    && python3 -c 'import torch; print("torch==" + torch.__version__)' > /tmp/torch.txt

COPY pyproject.toml setup.py ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip wheel . --no-deps -w /wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --build-constraint /tmp/torch.txt

FROM ${COURSE_IMAGE} AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libstdc++6 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=build /tmp/torch.txt /tmp/torch.txt

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install \
    -r /tmp/torch.txt --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install "pytest>=9.1.1" \
    && rm /tmp/torch.txt

COPY --from=build /wheels/ /wheels/

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install /wheels/*.whl \
    && python3 -m pip check \
    && rm -rf /wheels

COPY tests/ /opt/tests/

CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]
