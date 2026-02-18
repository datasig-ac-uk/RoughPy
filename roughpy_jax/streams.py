import typing

from typing import Protocol, TypeVar, Self, Callable

import numpy as np
import jax
import jax.numpy as jnp


from .intervals import Interval

LieT = TypeVar("LieT")
GroupT = TypeVar("GroupT")
StreamValueT = TypeVar("StreamValueT")

BasisLike = TypeVar("BasisLike")  ## TODO: replace with version from ops branch


@typing.runtime_checkable
class Stream(Protocol[LieT, GroupT]):
    """
    A stream is a description of the evolution of a system.

    A stream is any object that provides a (log-)signature over any query intervals.
    The signature is usually a group-like element of the free tensor algebra. More
    generally, the signature can be an element of a fairly arbitrary Lie group, such
    as a compact matrix group, and the log-signature belongs to the corresponding
    Lie Algebra.

    There are several, fundamentally different, types of stream, and generally these
    will be stacked to form a transformation pipeline. The streams at the root of
    such pipelines (the input) will often provide access to raw data, which are
    then transformed by, for example, solving a controlled differential equation.
    Other types of stream might compute signatures from deterministic functions or
    from stochastic sources like Brownian motion.

    This class is part of a system designed to model and process continuous data
    streams, allowing efficient calculation of both signatures and log signatures.
    The Stream interface is defined as a protocol, thus any implementation must be
    compliant with the method contracts defined herein.
    """

    @property
    def lie_basis(self) -> BasisLike:
        """
        A basis of the Lie algebra into which the stream is developed.
        """
        ...

    @property
    def group_basis(self) -> BasisLike:
        """
        A basis of the group into which the stream is developed.
        """
        ...

    @property
    def support(self) -> Interval:
        """
        The support interval for the stream.

        Outside of this interval, the signature should return the identity element.
        """
        ...

    def log_signature(self, interval: Interval) -> LieT:
        """
        Query the stream for the log signature over an interval.

        The log signature describes the evolution of the stream over an interval.
        This contains the same information as the signature, but is usually a more
        compressed representation.

        When this is the standard Lie algebra development, continuous functions
        on the underlying stream can be approximated uniformly by polynomials
        on the log signature.

        One should be able to query the stream over arbitrary intervals. This might
        include queries where the stream has no recorded changes, in which case the
        log signature is zero. (This includes the case where the query is outside
        the support of the stream.)

        :param interval: Query interval
        :return: A Lie element describing the stream over the interval.
        """
        ...

    def signature(self, interval: Interval) -> GroupT:
        """
        Query the stream for the signature over an interval.

        The signature describes the evolution of the stream over an interval.

        When this is the standard free tensor algebra development, continuous functions
        on the underlying stream can be approximated uniformly by linear functionals
        on the signature; that is, shuffle tensors.

        One should be able to query the stream over arbitrary intervals. This might
        include queries where the stream has no recorded changes, in which case the
        signature is identity. (This includes the case where the query is outside
        the support of the stream.)

        :param interval: Query interval
        :return: A Lie element describing the stream over the interval.
        """
        ...


@typing.runtime_checkable
class ValueStream(Protocol[LieT, GroupT, StreamValueT]):
    """
    A value stream tracks the value as well as the evolution.

    A value stream is a pair of a (increment) stream and base value. The stream value
    at any given parameter t is obtained by propagating the base value using the
    signature over from t_0 up to t.

    A very basic version of a ValueStream is a tensor-valued stream, where the
    value type is a free tensor and the propagation operation is left multiplication
    by the signature. More generally, this might involve the action of a linear
    projection of the signature.

    The basic operation on a value stream is `query`. This takes a query interval and
    produces a new value path. The increment stream of the new path is obtained by
    restricting to the query interval. The new base value is obtained by propagating
    the base value of the original stream using the signature up to the beginning
    of the query interval. (There is an equivalent formulation that uses terminal
    values and signature from the end of the query interval.)

    The other functions are there to get the stream, base value, and propagation
    function from the value stream. These are useful for inspecting the stream
    at the end of a pipeline.

    :ivar value_propagate_func: Callable that propagates the base value according
          to the signature.
    :type value_propagate_func: Callable[[GroupT, StreamValueT], StreamValueT]
    :ivar stream: The underlying increment stream.
    :type stream: Stream[LieT, GroupT]
    :ivar base_value: The value at the beginning (or end) of the stream.
    :type base_value: StreamValueT
    """

    @property
    def value_propagate_func(self) -> Callable[[GroupT, StreamValueT], StreamValueT]:
        """
        The function that propagates the base value according to the signature.
        """
        ...

    @property
    def stream(self) -> Stream[LieT, GroupT]:
        """
        The underlying increment stream.
        """
        ...

    @property
    def base_value(self) -> StreamValueT:
        """
        The value at the beginning (or end) of the stream.
        """
        ...

    def query(self, interval: Interval) -> Self:
        """
        Query the value stream over an interval.

        Query the value stream over an interval. The increment stream of the new path
        is obtained by restricting to the query interval. The new base value is
        obtained by propagating the base value of the original stream using the
        signature up to the beginning of the query interval. (There is an equivalent
        formulation that uses terminal values and signature from the end of the query
        interval.)

        :param interval: Query interval.
        :return: A new value stream, with an updated base value and restricted increment
                 stream.
        """
        ...
