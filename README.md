# PyTorch C++ Extension

A custom CPU tensor operation distributed as an installable Python wheel.
The implementation uses `CppExtension`, `BuildExtension`, and pybind11.

## Operation

**Python function:** `fse_ops.leaky_relu(x)`.

The operation keeps positive elements unchanged and multiplies every other
element by 0.1. Its exact pure-PyTorch specification is:

```python
torch.where(x > 0, x, 0.1 * x)
```

Input requirements:

- CPU device;
- `torch.float32` dtype;
- strided layout.

The output preserves the input shape and dtype. Scalars, empty tensors,
noncontiguous tensors, and leading batch dimensions are supported.
Unsupported devices, dtypes, and layouts raise `RuntimeError` through
`TORCH_CHECK` in C++. Autograd is not implemented.

```python
import torch
from fse_ops import leaky_relu

x = torch.tensor([-5.0, 0.5, 2.0], dtype=torch.float32)
print(leaky_relu(x))
# tensor([-0.5000, 0.5000, 2.0000])
```

## Requirements

- CPU-only Linux for the course verification;
- Python 3.11 or newer with venv support;
- a compiler supporting C++20;
- GNU Make.

Run the Python and Make commands below from the project directory, which
contains `pyproject.toml` and `setup.py`. For course verification, use the
course sandbox (`lab`). Run Docker commands on the host with the Docker daemon.

## Build and install the wheel

On a fresh checkout, create a build environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make wheel
make install
```

If `.venv` already exists, activate it instead of recreating it.
The environment must have pip; for a venv created without it, run
`python -m ensurepip --upgrade` once after activation.
`make setup` installs CPU PyTorch 2.14.1, setuptools, NumPy, pytest, and Ruff.
`make wheel` compiles the C++ extension and writes a Linux wheel to `dist/`.
It uses `--no-build-isolation` to build with the dependencies installed by
`make setup`, so it does not download another copy of torch for the build.
`make install` installs the wheel, replacing any earlier installation of
this package even if its version is unchanged.

After a C++ change, repeat `make wheel` and `make install` before testing.

## Test the installed wheel

```bash
make test
```

This copies `tests/` to a temporary directory outside the checkout and runs
pytest with the selected Python interpreter. The suite contains eight CPU
cases covering shape, dtype, exact values, batch size four, distinct
per-element outputs, and rejected dtypes. Expected result: `8 passed`.

For a separate installation check, keep the build environment activated and
create a fresh test environment with the same Python version:

```bash
test_env=$(mktemp -d /tmp/fse-test.XXXXXX)
python3 -m venv "$test_env"
make setup PYTHON="$test_env/bin/python"
make install PYTHON="$test_env/bin/python"
make test PYTHON="$test_env/bin/python"
rm -rf -- "$test_env"
```

The fresh environment receives the wheel from `dist/`; it does not use an
editable installation. Tests are again executed outside the source tree.

## Makefile targets

| Target | Action |
| --- | --- |
| `setup` | Install CPU torch, build and runtime dependencies, pytest, and Ruff into a venv |
| `wheel` | Build the wheel in `dist/`, replacing previous wheels there |
| `install` | Install the wheel and check dependency consistency |
| `test` | Run pytest outside the checkout against the installed package |
| `clean` | Remove `dist/`, `build/`, and project `__pycache__/` directories |

Targets use the active environment's `python3`. Override it with
`PYTHON=/path/to/venv/bin/python` when necessary. `clean` preserves Git history
and virtual environments.

## Smoke check

```bash
./tools/smoke.sh
```

The executable Bash script builds a wheel in `dist/`, verifies its Linux tag
with `grep`, installs CPU torch and the wheel into a second fresh venv, and
imports and calls the function from outside the checkout. It prints `OK`
only after all steps succeed and removes its temporary environments on exit.

## Docker

From the project directory on the host:

```bash
docker build -t fse-ops:0.1 .
docker run --rm fse-ops:0.1
```

The builder stage compiles the wheel from source. The final stage installs
CPU torch first, then the wheel and pytest, and runs
`python3 -m pytest -q /opt/tests`. Expected result: `8 passed`.
The compiler is not copied into the final stage. `.dockerignore` excludes
Git metadata, local environments, build outputs, and compiled extensions.

Both stages default to `ubuntu:24.04`, the base of the local course sandbox
recipes. The assignment does not specify the lab image tag. If a different
course image is supplied, select it with
`--build-arg COURSE_IMAGE=your-course-image` when building.

## CI and project note

[GitHub Actions](.github/workflows/ci.yml) runs on push and pull requests:
Ruff, wheel build, installation into a fresh CPU environment, and pytest
outside the checkout.

[NOTE.md](NOTE.md) records the real bug, its discovery and fix commit, an
unsupported input, and a large-input cProfile measurement with its command
and result. Git author identities are listed in [AUTHORS](AUTHORS).

## Authors

Git author identities matching the repository history are listed in
[AUTHORS](AUTHORS). Contact addresses and responsibilities follow.

Ivan Steksov (steksov.iv@phystech.edu, aka ivan.steksov@skoltech.edu)

Responsible for core function implementation, tests, CLI, environment setup, and maintaining a clean and consistent Git commit history.

Sultan Akhmetov (sultan.akhmetov@skoltech.ru)

Responsible for Docker configuration, Makefile, and smoke tests.
