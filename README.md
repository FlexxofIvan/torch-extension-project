# PyTorch C++ Extension

A custom CPU tensor operation implemented as a PyTorch C++ extension.

## Project Overview

This project implements a custom PyTorch operation in C++ and
distributes it as an installable Python wheel.

The project includes:
- A custom C++ CPU kernel.
- Python bindings for the operation.
- Unit tests using pytest.
- Automated builds and testing with GitHub Actions.
- A multi-stage Docker image.
- A smoke test for wheel installation and execution.

## Requirements

- Python 3.11
- C++20
- PyTorch (CPU)
- C++ compiler
- pytest
- ruff

## Installation

From the project directory, using Python 3.11 and a C++ compiler:

```bash
python3 -m venv /tmp/fse-build
/tmp/fse-build/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/fse-build/bin/pip wheel . --no-deps -w dist

python3 -m venv /tmp/fse-test
/tmp/fse-test/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/fse-test/bin/pip install dist/*.whl
```

## Usage

`fse_ops.leaky_relu(x)` keeps positive values and multiplies all other
values by 0.1. Its pure-PyTorch specification is:

```python
torch.where(x > 0, x, 0.1 * x)
```

The input must be a strided CPU tensor with dtype `torch.float32`.
Other devices, dtypes, and layouts raise `RuntimeError` from C++.
The output preserves the input shape and dtype, including empty and
noncontiguous inputs. Autograd is not implemented.

```python
import torch
from fse_ops import leaky_relu

print(leaky_relu(torch.tensor([-5.0, 0.5, 2.0])))
# tensor([-0.5000, 0.5000, 2.0000])
```

## Testing

After installing the wheel as above, run the tests outside the source tree:

```bash
/tmp/fse-test/bin/pip install pytest
test_dir=$(mktemp -d /tmp/fse-tests.XXXXXX)
cp -r tests "$test_dir/tests"
(cd "$test_dir" && /tmp/fse-test/bin/python -m pytest -q tests)
```

## Profiling

With the package installed in `.venv`, run from the project directory:

```bash
.venv/bin/python -m cProfile -o /tmp/fse-leaky-relu.prof tools/profile_op.py
.venv/bin/python -c 'import pstats; pstats.Stats("/tmp/fse-leaky-relu.prof").strip_dirs().sort_stats("cumulative").print_stats("run_calls|leaky_relu")'
```

The script creates a contiguous CPU `float32` tensor with 10,000,000
elements spanning -5 to 5, performs one warm-up call, then runs 20 calls
with one PyTorch thread. Each call allocates a new output tensor.

Measured on 2026-10-21, Linux x86_64, AMD Ryzen 5 5600H,
Python 3.11.14, PyTorch 2.14.1+cu130 (CPU tensors):

```text
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     1    0.065    0.065    0.860    0.860 profile_op.py:10(run_calls)
    21    0.842    0.040    0.842    0.040 {built-in method fse_ops._C.leaky_relu}
```

The 20-call loop took 0.860 seconds cumulatively, about 43 ms per call,
including output allocation and release. Its interval excludes imports,
input creation, and warm-up. The extension row includes the warm-up,
hence 21 calls. These are local profiling results, not a comparison
against PyTorch's built-in operation; timings vary with the environment.
`cProfile` reports the extension call as a whole, without a breakdown
of its internal C++ work.

## Docker

To be added.

## Authors

To be added.
