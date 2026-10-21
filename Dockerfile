FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel 

RUN python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml setup.py ./
COPY src/ ./src/

RUN python3 -m pip wheel . --no-deps -w /wheels

FROM python:3.11-slim AS runtime

WORKDIR /opt

RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir pytest

COPY --from=build /wheels/ /wheels/

RUN python3 -m pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY tests/ /opt/tests/

CMD ["python3", "-m", "pytest", "-q", "/opt/tests"]