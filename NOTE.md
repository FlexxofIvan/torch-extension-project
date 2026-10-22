# Project note

## Bug: lazy negative views

The regression test `test_exact_values[negative-view]` failed with an
`AssertionError`: the extension disagreed with
`torch.where(x > 0, x, 0.1 * x)`. The input had a lazy negative bit, so its
logical values were the negation of the underlying storage. For example,
`torch._neg_view(torch.tensor([-1.0, 2.0]))` represents `[1.0, -2.0]`,
but the original kernel read `[-1.0, 2.0]` through its accessor and returned
`[-0.1, 2.0]` instead of `[1.0, -0.2]`.

Commit `4120479` fixes this by replacing `x.reshape({-1})` with
`x.resolve_neg().reshape({-1})` before accessing the data. After rebuilding
the extension, the regression test passed; the installed-wheel suite also
passed all eight cases. The focused test can be run with:

```bash
.venv/bin/python -m pytest -v 'tests/test_op.py::test_exact_values[negative-view]'
```

## Unsupported case

The kernel accepts CPU `float32` tensors only. For example,
`leaky_relu(torch.tensor([1.0, 2.0], dtype=torch.float64))` raises
`RuntimeError: leaky_relu expects a float32 tensor` from the C++
`TORCH_CHECK`; it does not silently cast the input.

## Large-input measurement

From the project directory, with the package installed in `.venv`:

```bash
.venv/bin/python -m cProfile -o /tmp/fse-leaky-relu.prof tools/profile_op.py
.venv/bin/python -c 'import pstats; pstats.Stats("/tmp/fse-leaky-relu.prof").strip_dirs().sort_stats("cumulative").print_stats("run_calls|leaky_relu")'
```

On 2026-10-21, on an AMD Ryzen 5 5600H with Linux x86_64, Python 3.11.14
and PyTorch 2.14.1+cu130 using CPU tensors, `run_calls` printed a cumulative
time of **0.860 seconds for 20 calls**, about **43 ms per call**. The input
was a contiguous `float32` tensor of 10,000,000 values spanning -5 to 5,
with one PyTorch thread. The measured loop includes output allocation and
release, but excludes imports, input creation, and one warm-up call. The
extension's separate profile row includes the warm-up, so it reports
21 calls. This is a local measurement, not a speed comparison with
PyTorch; `cProfile` does not break down the kernel's internal C++ work.
