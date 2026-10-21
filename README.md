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

## Docker

To be added.

## Authors

To be added.
