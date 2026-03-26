from __future__ import annotations

from abc import ABC as AbstractBaseClass
import enum
import math
import numbers
import typing
from dataclasses import dataclass
from typing import Generic, Protocol, Self, TypeVar

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

RealT = TypeVar("RealT")


class IntervalType(enum.IntEnum):
    ClOpen = 0
    OpenCl = 1


@typing.runtime_checkable
class Interval(Protocol):
    """
    Representation of an interval with an unspecified type for mathematical
    computations or range definitions. This interface outlines the required
    methods for retrieving key properties of an interval.
    The purpose of this protocol is to define a constraint for other classes
    or structures that aim to represent intervals and their specific types,
    lower bounds (infimum), and upper bounds (supremum). Classes that adhere
    to this protocol should implement the specified methods to be considered
    compatible.
    :ivar interval_type: Type information of the interval. This attribute
        specifies the nature or classification of the interval, such as
        open, closed, or partially open/closed.
    :type interval_type: IntervalType
    :ivar inf: The lower bound (infimum) of the interval. This attribute
        represents the smallest value contained within the interval.
    :type inf: float
    :ivar sup: The upper bound (supremum) of the interval. This attribute
        represents the largest value contained within the interval.
    :type sup: float
    """

    @property
    def interval_type(self) -> IntervalType: ...

    @property
    def inf(self) -> Array: ...

    @property
    def sup(self) -> Array: ...

    @property
    def length(self) -> Array: ...


class BaseInterval(AbstractBaseClass):
    # TODO: These don't need to be in a class, just have module-level functions for str, length, and intersection that
    # take Intervals as arguments. The only reason to have these in a class is if we want to use inheritance to share
    # code between different Interval implementations.

    @staticmethod
    def to_string(interval: Interval) -> str:
        reprs = {
            IntervalType.ClOpen: "[{}, {})",
            IntervalType.OpenCl: "({}, {}]",
        }
        return reprs[interval.interval_type].format(interval.inf, interval.sup)

    @staticmethod
    def length(interval: Interval) -> Array:
        """
        Calculate the length of the interval.
        :return: The length of the interval, calculated as sup - inf.
        :rtype: float
        """
        return jnp.maximum(0.0, jnp.asarray(interval.sup) - jnp.asarray(interval.inf))


def intersection(
    left_interval: Interval,
    right_interval: Interval,
) -> RealInterval:
    """
    Calculate the intersection of this interval with another interval.
    :param other: The other interval to intersect with.
    :type other: Interval
    :return: A new Interval representing the intersection, or None if there is no intersection.
    :rtype: typing.Optional[RealInterval]
    """
    if not isinstance(left_interval, Interval) or not isinstance(
        right_interval, Interval
    ):
        raise TypeError("Both arguments must be of type Interval")

    if left_interval.interval_type != right_interval.interval_type:
        raise TypeError("Both intervals must be of the same IntervalType")

    new_inf = jnp.maximum(left_interval.inf, right_interval.inf)
    new_sup = jnp.minimum(left_interval.sup, right_interval.sup)
    interval_type = left_interval.interval_type

    # Intersections are defined by bounds, but not every Interval implementation
    # can be constructed from (inf, sup, interval_type) (e.g. DyadicInterval).
    # Use a canonical RealInterval result to avoid incorrect reconstruction.
    return RealInterval(_inf=new_inf, _sup=new_sup, _interval_type=interval_type)


@dataclass(frozen=True)
class Dyadic:
    """
    Represents a dyadic number in mathematics.
    Dyadic numbers are numbers of the form k * (2^-n), where k is an integer and
    n is a non-negative integer.
    :ivar k: Integer component of the dyadic number.
    :type k: int
    :ivar n: Exponent of 2 in the dyadic number.
    :type n: int
    """

    k: int
    n: int

    def __post_init__(self) -> None:
        # NOTE: This whole method feels a little unnecessary since the dataclass will already enforce that k and n are
        # integers but it does allow for some flexibility in accepting float inputs that are integer-valued, which could
        # be convenient in some cases. It also allows us to provide more informative error messages if the inputs are
        # not valid.

        k = self.k
        n = self.n

        if isinstance(k, numbers.Integral):
            k_int = int(k)
        elif isinstance(k, float) and k.is_integer():
            k_int = int(k)
        else:
            raise TypeError(
                f"Dyadic.k must be an integer-like value, got {type(k).__name__}: {k!r}"
            )

        if isinstance(n, numbers.Integral):
            n_int = int(n)
        elif isinstance(n, float) and n.is_integer():
            n_int = int(n)
        else:
            raise TypeError(
                f"Dyadic.n must be an integer-like value, got {type(n).__name__}: {n!r}"
            )

        if n_int < 0:
            raise ValueError(f"Dyadic.n must be non-negative, got {n_int}")

        object.__setattr__(self, "k", k_int)
        object.__setattr__(self, "n", n_int)

    def __float__(self) -> float:
        return math.ldexp(self.k, -self.n)

    def __jax_array__(self) -> Array:
        return jnp.array(math.ldexp(self.k, -self.n))


@dataclass(frozen=True)
class DyadicInterval(Dyadic):
    """
    This subclass represents a dyadic interval, and therefore conforms to the
    Interval protocol. Crucially this is a subclass of Dyadic becuase its
    endpoints (and width) are fully defined by the k and n parameters of Dyadic
    and whether the interval is closed or open on either end.
    """

    _interval_type: IntervalType

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> Array:
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return jnp.ldexp(k, -self.n)

    @property
    def sup(self) -> Array:
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return jnp.ldexp(k, -self.n)

    @property
    def length(self) -> Array:
        return BaseInterval.length(self)


@dataclass(frozen=True)
class RealInterval:
    """
    Represents a real interval with specified bounds and interval type.
    This class is used to define and represent a mathematical interval in the real number
    line. It includes the lower bound, upper bound, and the type of interval (e.g., open,
    closed). The class is immutable and all fields are frozen upon initialization.
    :ivar interval_type: Indicates the type of interval (e.g., open, closed).
    :type interval_type: IntervalType
    :ivar inf: The lower bound of the interval.
    :type inf: RealT
    :ivar sup: The upper bound of the interval.
    :type sup: RealT
    """

    _inf: float | Array
    _sup: float | Array
    _interval_type: IntervalType

    def __str__(self) -> str:
        return BaseInterval.to_string(self)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> float | Array:
        return self._inf

    @property
    def sup(self) -> float | Array:
        return self._sup

    @property
    def length(self) -> float | Array:
        return BaseInterval.length(self)


RealInterval = jax.tree_util.register_dataclass(
    RealInterval,
    data_fields=["_inf", "_sup"],
    meta_fields=["_interval_type"],
)


@dataclass(frozen=True)
class Partition:
    _endpoints: list
    _interval_type: IntervalType

    def __len__(self) -> int:
        return len(self._endpoints) - 1

    def __str__(self) -> str:
        return BaseInterval.to_string(self)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> Array:
        return jnp.asarray(self._endpoints[0])

    @property
    def sup(self) -> Array:
        return jnp.asarray(self._endpoints[-1])

    @property
    def length(self) -> Array:
        return BaseInterval.length(self)

    def to_intervals(self) -> list[RealInterval]:
        """
        Convert the partition into a list of RealIntervals corresponding to
        the subintervals defined by the partition.
        :param partition: The Partition to convert.
        :type partition: Partition
        :return: A list of RealIntervals representing the subintervals of the partition.
        :rtype: list[RealInterval]
        """
        return [
            RealInterval(
                _inf=self._endpoints[i],
                _sup=self._endpoints[i + 1],
                _interval_type=self.interval_type,
            )
            for i in range(len(self._endpoints) - 1)
        ]


Partition = jax.tree_util.register_dataclass(
    Partition,
    data_fields=["_endpoints"],
    meta_fields=["_interval_type"],
)
