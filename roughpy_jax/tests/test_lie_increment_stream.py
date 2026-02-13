import jax.numpy as jnp
import pytest

from roughpy_jax.algebra import LieBasis, TensorBasis
from roughpy_jax.intervals import IntervalType, RealInterval
import roughpy_jax.streams.lie_increment_stream as lis_mod
from roughpy_jax.streams.lie_increment_stream import LieIncrementStream


def test_from_stream_rejects_nonpositive_resolution():
    class DummyStream:
        lie_basis = LieBasis(width=1, depth=1)
        group_basis = TensorBasis(width=1, depth=1)
        support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

    with pytest.raises(ValueError, match="resolution must be positive"):
        LieIncrementStream.from_stream(DummyStream(), resolution=0)


def test_from_stream_uses_stream_dyadic_cache_provider():
    class DummyStream:
        def __init__(self):
            self.lie_basis = LieBasis(width=1, depth=1)
            self.group_basis = TensorBasis(width=1, depth=1)
            self.support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        def __dyadic_cache__(self, resolution: int):
            return jnp.zeros((1 << (resolution + 1), self.lie_basis.size()), dtype=jnp.float32)

    src = DummyStream()
    result = LieIncrementStream.from_stream(src, resolution=4)

    assert isinstance(result, LieIncrementStream)
    assert result.resolution == 4
    assert result.support == src.support
    assert result.lie_basis == src.lie_basis
    assert result.group_basis == src.group_basis
    assert result.__base_stream__ is src


def test_from_stream_falls_back_to_stream_to_cache(monkeypatch):
    class DummyStream:
        def __init__(self):
            self.lie_basis = LieBasis(width=1, depth=1)
            self.group_basis = TensorBasis(width=1, depth=1)
            self.support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

    src = DummyStream()
    captured = {}

    def fake_stream_to_cache(stream, resolution, interval_type=IntervalType.ClOpen):
        captured["stream"] = stream
        captured["resolution"] = resolution
        captured["interval_type"] = interval_type
        return jnp.zeros((1 << (resolution + 1), src.lie_basis.size()), dtype=jnp.float32)

    monkeypatch.setattr(LieIncrementStream, "_stream_to_cache", staticmethod(fake_stream_to_cache))

    result = LieIncrementStream.from_stream(src, resolution=3)
    assert isinstance(result, LieIncrementStream)
    assert result.resolution == 3
    assert captured["stream"] is src
    assert captured["resolution"] == 3
    assert captured["interval_type"] == IntervalType.ClOpen


def test_from_increments_wires_builder_and_cache_extension(monkeypatch):
    captured = {"ks": []}

    def fake_build_base_entry(
        k,
        k_arrays,
        data,
        data_basis,
        cache_basis,
        tensor_basis,
    ):
        captured["ks"].append(k)
        captured["k_arrays"] = k_arrays
        captured["data"] = data
        captured["data_basis"] = data_basis
        captured["cache_basis"] = cache_basis
        captured["tensor_basis"] = tensor_basis
        return k

    def fake_extend_cache_from_base(base, resolution, cache_basis):
        captured["base"] = list(base)
        captured["resolution"] = resolution
        captured["extended_basis"] = cache_basis
        return jnp.zeros((1 << (resolution + 1), cache_basis.size()), dtype=jnp.float32)

    monkeypatch.setattr(lis_mod, "_build_base_entry", fake_build_base_entry)
    monkeypatch.setattr(lis_mod, "_extend_cache_from_base", fake_extend_cache_from_base)

    timestamps = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    data = jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float32)

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=2,
        input_data_basis=None,
        lie_basis=LieBasis(width=1, depth=2),
        interval_type=IntervalType.ClOpen,
    )

    assert isinstance(stream, LieIncrementStream)
    assert stream.resolution == 2
    assert stream.support.interval_type == IntervalType.ClOpen
    assert captured["ks"] == [0, 1, 2, 3]
    assert captured["base"] == [0, 1, 2, 3]
    assert captured["resolution"] == 2
    assert len(captured["k_arrays"]) == 1
    assert captured["k_arrays"][0].dtype == jnp.dtype("int32")
