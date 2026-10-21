#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"

BUILD_VENV="$(mktemp -d /tmp/fse-build.XXXXXX)"
TEST_VENV="$(mktemp -d /tmp/fse-test.XXXXXX)"

cleanup() {
    rm -rf "$BUILD_VENV" "$TEST_VENV"
}
trap cleanup EXIT

# Build the wheel in a dedicated environment.
python3 -m venv "$BUILD_VENV" &&
"$BUILD_VENV/bin/python" -m pip install \
    torch --index-url https://download.pytorch.org/whl/cpu &&
mkdir -p "$DIST_DIR" &&
rm -f "$DIST_DIR"/*.whl &&
"$BUILD_VENV/bin/python" -m pip wheel \
    "$ROOT_DIR" --no-deps -w "$DIST_DIR"

# Find the wheel and verify that it is Linux-specific.
WHEEL="$(find "$DIST_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)"

if [[ -z "$WHEEL" ]]; then
    echo "ERROR: no wheel was produced" >&2
    exit 1
fi

if [[ "$(basename "$WHEEL")" != *linux* ]]; then
    echo "ERROR: wheel is not Linux-specific: $(basename "$WHEEL")" >&2
    exit 1
fi

# Install the wheel in a separate, fresh environment.
python3 -m venv "$TEST_VENV" &&
"$TEST_VENV/bin/python" -m pip install \
    torch --index-url https://download.pytorch.org/whl/cpu &&
"$TEST_VENV/bin/python" -m pip install "$WHEEL"

# Import and test the installed package outside the source tree.
cd /tmp &&
"$TEST_VENV/bin/python" - << 'EOF'
import torch
from fse_ops import leaky_relu

x = torch.tensor([-5.0, 0.5, 2.0])
expected = torch.where(x > 0, x, 0.1 * x)
result = leaky_relu(x)

assert torch.equal(result, expected), (
    f"Unexpected output: {result}; expected {expected}"
)
EOF
