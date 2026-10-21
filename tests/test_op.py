import pytest
import torch

from fse_ops import leaky_relu


@pytest.mark.parametrize(
    "x",
    [torch.arange(12, dtype=torch.float32).reshape(3, 4).T, torch.empty(0, 3)],
    ids=["noncontiguous", "empty"],
)
def test_output_shape(x):
    assert leaky_relu(x).shape == x.shape


def test_output_dtype():
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32)
    assert leaky_relu(x).dtype == torch.float32


@pytest.mark.parametrize("batched", [False, True], ids=["negative-view", "batch-four"])
def test_exact_values(batched):
    x = torch.tensor([-3.25, -0.5, 0.0, 0.25, 2.75], dtype=torch.float32)
    if batched:
        x = x.repeat(4, 1)
    else:
        # Keep the logical values, but store their negation with a lazy negative bit.
        x = torch._neg_view(-x)
        assert x.is_neg()
    expected = torch.where(x > 0, x, 0.1 * x)
    assert torch.equal(leaky_relu(x), expected)


def test_distinct_values():
    x = torch.tensor([-5.0, 0.5, 2.0], dtype=torch.float32)
    expected = torch.where(x > 0, x, 0.1 * x)
    result = leaky_relu(x)
    assert torch.equal(result, expected)
    assert not torch.equal(result, x)
    assert result.unique().numel() == x.numel()


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_invalid_dtype(dtype):
    x = torch.tensor([1, 2, 3], dtype=dtype)
    with pytest.raises(RuntimeError, match="expects a float32 tensor"):
        leaky_relu(x)
