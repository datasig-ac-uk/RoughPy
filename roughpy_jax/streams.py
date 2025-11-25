import abc
import typing

from dataclasses import dataclass
from typing import Protocol, TypeVar, TypeAlias, Self, Generic

import jax
import jax.numpy as jnp

from roughpy import compute as rpc

from roughpy_jax.intervals import Partition, Interval
from roughpy_jax.algebra import DenseFreeTensor, DenseLie, TensorBasis, LieBasis, lie_to_tensor, ft_exp


TensorT = TypeVar("TensorT")
LieT = TypeVar("LieT")
StreamValueT = TypeVar("StreamValueT")


@typing.runtime_checkable
class Stream(Protocol[LieT, TensorT]):
    """
    Represents a Stream interface that operates with specific mathematical constructs
    such as Lie groups and Tensors, and provides functionality to compute signatures
    over specified intervals.

    This class is part of a system designed to model and process continuous data
    streams, allowing efficient calculation of both signatures and log signatures.
    The Stream interface is defined as a protocol, thus any implementation must be
    compliant with the method contracts defined herein.

    """

    def lie_basis(self) -> rpc.TensorBasis:
        ...

    def tensor_basis(self) -> rpc.TensorBasis:
        ...

    def log_signature(self, interval: Interval) -> LieT:
        ...

    def signature(self, interval: Interval) -> TensorT:
        ...




@typing.runtime_checkable
class ValueStream(Protocol[LieT, TensorT, StreamValueT]):

    def stream(self) -> Stream[LieT, TensorT]:
        ...

    def query(self, interval: Interval) -> Self:
        ...

    def base_value(self) -> StreamValueT:
        ...



@dataclass(frozen=True)
class DyadicCachedTickStream(Generic[LieT, TensorT]):

    # The Lie increment dyadic cache data
    _data: LieT

    # The maximum dyadic resolution held in the dyadic cache
    _resolution: int

    # basis for the Tensor algebra in which signatures take values
    _tensor_basis: TensorBasis

    # Basis for the Lie algebra in which log signatures take values
    _lie_basis: LieBasis


    def tensor_basis(self) -> TensorBasis:
        return self._tensor_basis

    def lie_basis(self) -> LieBasis:
        return self._lie_basis

    def signature(self, interval: Interval) -> TensorT:
        ...

    def log_signature(self, interval: Interval) -> LieT:
        ...






def to_dyadic_cached_tick_stream(stream: Stream[LieT, TensorT], resolution: int) -> DyadicCachedTickStream[LieT, TensorT]:
    """
    Converts a `Stream` into a `DenseDyadicCachedTickStream` format using a specified
    resolution. This operation enables better data organization and caching of
    time-series data aligned with a dyadic time interval. This function is useful
    for applications where efficient and structured access to high-frequency
    time-series data is required.

    :param stream: The input time-series stream to be converted.
    :type stream: Stream
    :param resolution: The resolution to use for converting the stream, represented
        as an integer defining the dyadic time interval.
    :type resolution: int
    :return: A `DenseDyadicCachedTickStream` object, which is a cached
        representation of the input stream aligned to dyadic resolution.
    :rtype: DenseDyadicCachedTickStream
    """
    ...




class PiecewiseAbelianStream(Generic[LieT, TensorT]):
    ...



def to_piecewise_abelian_stream(stream: Stream[LieT, TensorT], partition: Partition) -> PiecewiseAbelianStream[LieT, TensorT]:
    """
    Converts a given stream into a piecewise Abelian stream representation using the provided partition.

    This function processes the provided ``stream`` and utilizes the specified ``partition`` to
    construct a ``DensePiecewiseAbelianStream``. The resulting object is a representation that
    segments the input stream into parts based on the partition criteria, encoding the Abelian
    properties in a dense format.

    :param stream: The input stream to be converted.
    :type stream: Stream
    :param partition: The partition object used to segment the stream into its piecewise parts.
    :type partition: Partition
    :return: A new DensePiecewiseAbelianStream created from the given stream and partition.
    :rtype: DensePiecewiseAbelianStream
    """
    ...





class TensorValuedStream(Generic[LieT, TensorT]):
    _increment_stream: Stream[LieT, TensorT]
    _base_value: TensorT

    def stream(self) -> Stream[LieT, TensorT]:
        return self._increment_stream

    def query(self, interval: Interval) -> TensorT:
        ...

    def base_value(self) -> TensorT:
        return self._base_value






class BaseStream(Generic[LieT, TensorT], metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute_increment(self, inf: float, sup: float) -> LieT:
        ...



    @abc.abstractmethod
    def lie_basis(self) -> LieBasis:
        ...

    @abc.abstractmethod
    def tensor_basis(self) -> TensorBasis:
        ...

    def log_signature(self, interval: Interval) -> LieT:

        # Replace this by a dyadic inspection perhaps
        val = self.compute_increment(interval.inf(), interval.sup())
        return val

    def signature(self, interval: Interval) -> TensorT:
        ls_val = self.log_signature(interval)
        return ft_exp(lie_to_tensor(ls_val), out_basis=self.tensor_basis())

