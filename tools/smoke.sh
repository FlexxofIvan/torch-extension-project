#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"

SMOKE_DIR="$(mktemp -d /tmp/fse-smoke.XXXXXX)"
BUILD_VENV="$SMOKE_DIR/build-venv"
TEST_VENV="$SMOKE_DIR/test-venv"

cleanup() {
    rm -rf "$SMOKE_DIR"
}
trap cleanup EXIT

# Build the wheel in a dedicated environment.
python3 -m venv "$BUILD_VENV" &&
"$BUILD_VENV/bin/python" -m pip install --upgrade pip &&
"$BUILD_VENV/bin/python" -m pip install \
    "torch>=2.14.1" --index-url https://download.pytorch.org/whl/cpu &&
"$BUILD_VENV/bin/python" -c \
    'from importlib.metadata import version; print("torch==" + version("torch"))' \
    > "$SMOKE_DIR/torch.txt" &&
mkdir -p "$DIST_DIR" &&
rm -f "$DIST_DIR"/*.whl &&
"$BUILD_VENV/bin/python" -m pip wheel \
    "$ROOT_DIR" --no-deps -w "$DIST_DIR" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --build-constraint "$SMOKE_DIR/torch.txt" || exit $?

# Find the wheel and verify that it is Linux-specific.
WHEEL="$(find "$DIST_DIR" -maxdepth 1 -type f -name '*.whl' -print -quit)"

if [[ -z "$WHEEL" ]]; then
    echo "ERROR: no wheel was produced" >&2
    exit 1
fi

basename "$WHEEL" | grep linux || exit $?

# Install the wheel in a separate, fresh environment.
python3 -m venv "$TEST_VENV" &&
"$TEST_VENV/bin/python" -m pip install \
    -r "$SMOKE_DIR/torch.txt" --index-url https://download.pytorch.org/whl/cpu &&
"$TEST_VENV/bin/python" -m pip install "$WHEEL" &&
mkdir "$SMOKE_DIR/run" || exit $?

# Import and test the installed package outside the source tree.
cd "$SMOKE_DIR/run" &&
"$TEST_VENV/bin/python" -I - << 'EOF' || exit $?
import torch
from fse_ops import leaky_relu

x = torch.tensor([-5.0, 0.5, 2.0])
expected = torch.where(x > 0, x, 0.1 * x)
result = leaky_relu(x)

assert torch.equal(result, expected), (
    f"Unexpected output: {result}; expected {expected}"
)
EOF

echo OK
