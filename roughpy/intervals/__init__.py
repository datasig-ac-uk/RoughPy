
import enum
import math
import typing

from dataclasses import dataclass
from typing import TypeVar, Protocol, Generic, Any

import numpy as np

ParamT = TypeVar("ParamT")


class IntervalType(enum.IntEnum):
    ClOpen = 1
    OpenCl = 2


Clopen = IntervalType.ClOpen
Opencl = IntervalType.OpenCl


@typing.runtime_checkable
class Interval(Protocol[ParamT]):
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
    def interval_type(self) -> IntervalType:
        ...

    @property
    def inf(self) -> ParamT:
        ...

    @property
    def sup(self) -> ParamT:
        ...


def _interval_contains_interval(parent: Interval, child: Interval) -> bool:
    tp = type(parent.inf)

    oi = tp(child.inf)
    os = tp(child.sup)

    if parent.interval_type == child.interval_type:
        return parent.inf <= oi <= os <= parent.sup
    elif parent.interval_type == IntervalType.ClOpen:
        return parent.inf <= oi <= os < parent.sup
    else:
        return parent.inf < oi <= os <= parent.sup

def _interval_contains_param(parent: Interval[ParamT], child: Any) -> bool:
    tp = type(parent.inf)

    param = tp(child)

    if parent.interval_type == IntervalType.ClOpen:
        return parent.inf <= param < parent.sup

    return parent.inf < param <= parent.sup



def _interval_contains(parent: Interval[ParamT], child: Any) -> bool:
    if isinstance(child, Interval):
        return _interval_contains_interval(parent, child)

    return _interval_contains_param(parent, child)

def _interval_repr(interval: Interval[ParamT]) -> str:
    return f"Interval(inf={interval.inf}, sup={interval.sup}, type={interval.interval_type.name})"

def _interval_str(interval: Interval[ParamT]) -> str:
    if interval.interval_type == IntervalType.ClOpen:
        return f"[{interval.inf}, {interval.sup})"

    return f"({interval.inf}, {interval.sup}]"


def _interval_intersects_with(i1: Interval[ParamT], i2: Interval[ParamT]) -> bool:
    # inf and sup might not be trivial to compute
    i1_inf = i1.inf
    i1_sup = i1.sup

    i2_inf = i2.inf
    i2_sup = i2.sup

    # empty case
    if i1_inf == i1_sup or i2_inf == i2_sup:
        return False

    m_inf = max(i1_inf, i2_inf)
    m_sup = min(i1_sup, i2_sup)

    if m_inf < m_sup:
        return True

    if m_inf == m_sup:
        return _interval_contains_param(i2, m_inf) and _interval_contains_param(i2, m_inf)


    return False


def _real_interval_intersection(i1: Interval[ParamT], i2: Interval[ParamT]) -> RealInterval[ParamT]:
    inf = max(i1.inf, i2.inf)
    sup = min(i1.sup, i2.sup)

    ## correct the empty intervals
    if sup < inf:
        sup = inf

    return RealInterval(inf, sup, i1.interval_type)




def _interval_included_end(interval: Interval[ParamT]) -> ParamT:
    return interval.inf if interval.interval_type == IntervalType.ClOpen else interval.sup

def _interval_excluded_end(interval: Interval[ParamT]) -> ParamT:
    return interval.sup if interval.interval_type == IntervalType.ClOpen else interval.inf

def _interval_class(cls):

    if not hasattr(cls, "contains"):
        cls.contains = _interval_contains

    if not hasattr(cls, "__repr__"):
        cls.__repr__ = _interval_repr

    if not hasattr(cls, "__str__"):
        cls.__str__ = _interval_str

    if not hasattr(cls, "included_end"):
        cls.included_end = _interval_included_end

    if not hasattr(cls, "excluded_end"):
        cls.excluded_end = _interval_included_end



    return cls



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

    def __float__(self) -> float:
        return math.ldexp(self.k, -self.n)

    def dyadic_equals(self, other: Dyadic) -> bool:
        return self.n == other.n and self.k == other.k

    def rational_equals(self, other: Dyadic) -> bool:
        """
        Determine if two dyadics are equal as rational numbers.

        :param other: Another Dyadic object to compare with the current instance.
                      Must be an instance of the Dyadic class.
        :type other: Dyadic
        :return: True if the two Dyadic numbers are equal as rationals, otherwise False.
        :rtype: bool
        """
        k1 = self.k
        k2 = other.k

        if k1 == 0:
            return k2 == 0
        if k2 == 0:
            return False

        n = self.n - other.n

        if n < 0:
            return (k1 << -n) == k2
        else:
            return k1 == (k2 << n)





@_interval_class
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
    def dyadic_inf(self) -> Dyadic:
        return Dyadic(self.k if self._interval_type == IntervalType.ClOpen else self.k - 1, self.n)

    @property
    def dyadic_sup(self) -> Dyadic:
        return Dyadic(self.k + 1 if self._interval_type == IntervalType.ClOpen else self.k, self.n)

    @property
    def inf(self) -> float:
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return math.ldexp(self.k, -self.n)

    @property
    def sup(self) -> float:
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return math.ldexp(self.k + 1, -self.n)

    def __interval_intersection__(self, other: Interval) -> Interval:
        if not isinstance(other, DyadicInterval):
            return _real_interval_intersection(self, other)

        if self._interval_type != other._interval_type:
            raise ValueError("Cannot intersect intervals with different interval types")

        itype = self._interval_type
        is_clopen = itype == IntervalType.ClOpen

        if self.n <= other.n:
            smaller_k = self.k
            smaller_n = self.n
            larger_k = other.k
            larger_n = self.n
        else:
            smaller_k = other.k
            smaller_n = other.n
            larger_k = self.k
            larger_n = self.n

        inf_k = larger_k if is_clopen else larger_k - 1
        sup_k = larger_k + 1 if is_clopen else larger_k

        shift = larger_n - smaller_n
        inf_k <<= shift
        sup_k <<= shift

        if ((is_clopen and inf_k <= smaller_k < sup_k)
                or (not is_clopen and inf_k < smaller_k <= sup_k)):
            return DyadicInterval(smaller_k, smaller_n, itype)

        x = float(self)
        return RealInterval(x, x, itype)





    def divide(self) -> tuple[DyadicInterval, DyadicInterval]:
        """Divides the current dyadic interval into two equal parts.

        :return: A tuple containing two dyadic intervals that together cover the original interval.
        """
        itype = self.interval_type
        k = self.k
        n = self.n + 1

        if itype == IntervalType.ClOpen:
            return (DyadicInterval(k, n, itype),
                    DyadicInterval(k+1, n, itype))

        return (DyadicInterval(k-1, n, itype),
                DyadicInterval(k, n, itype))


    def shrink_to_contained_end(self, factor: int = 1) -> DyadicInterval:
        """
        Shrinks the current interval to the contained end by modifying its end point.

        This method creates a new DyadicInterval instance where the end of the current
        interval is adjusted by the specified factor. The start of the interval remains
        unchanged. The resulting interval retains the same interval type as the original one.

        :param factor: The offset added to the endpoint of the current interval.
                       Defaults to 1.
        :type factor: int
        :return: A new DyadicInterval instance with the adjusted endpoint.
        :rtype: DyadicInterval
        """
        return DyadicInterval(self.k, self.n + factor  , self.interval_type)

    def shrink_to_omitted_end(self) -> DyadicInterval:
        """
        Shrink the current interval to the omitted endpoint and return the resulting dyadic interval.

        This method computes a new interval by adjusting the left or right endpoint based
        on the type of interval (closed or open). The new interval's size is shifted by one level
        of refinement, corresponding to the binary scale of the dyadic interval.

        :return: A new DyadicInterval object computed based on the omitted endpoint.
        :rtype: DyadicInterval
        """
        return DyadicInterval(
            (self.k + 1) if self._interval_type == IntervalType.ClOpen else (self.k - 1),
            self.n + 1, self.interval_type)

    def flip_interval(self) -> DyadicInterval:
        """
        Flip the current dyadic interval to its adjacent interval based on the interval
        type and the parity of the `k` attribute.

        :return: A new dyadic interval, which is one step higher or lower than the
                 current interval based on the interval type and position parity.
        :rtype: DyadicInterval
        """
        unit = 1 if self.interval_type == IntervalType.ClOpen else -1
        if self.k & 1 == 0:
            return DyadicInterval(self.k + unit, self.n, self.interval_type)
        else:
            return DyadicInterval(self.k - unit, self.n, self.interval_type)


@_interval_class
@dataclass(frozen=True)
class RealInterval(Generic[ParamT]):
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
    inf: ParamT
    sup: ParamT
    interval_type: IntervalType


@_interval_class
@dataclass(frozen=True)
class Partition(Generic[ParamT]):
    _endpoints: list[ParamT]
    _interval_type: IntervalType

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> ParamT:
        return self._endpoints[0]

    @property
    def sup(self) -> ParamT:
        return self._endpoints[-1]






def intersection(i1: Interval[ParamT], i2: Interval[ParamT]) -> Interval[ParamT]:
    """
    Determines the intersection of two intervals. The intervals are intersected
    by finding the maximum of their lower bounds and the minimum of their upper
    bounds.

    If the intervals are of different types, a ``ValueError`` will be raised.

    :param i1: The first interval for the intersection operation.
    :type i1: Interval[ParamT]
    :param i2: The second interval for the intersection operation.
    :type i2: Interval[ParamT]
    :return: A new interval representing the intersection of the two input intervals.
    :rtype: Interval[ParamT]
    :raises ValueError: If the two intervals have mismatching types and cannot
        be intersected.
    """
    if i2.interval_type != i1.interval_type:
        raise ValueError(f"Cannot intersect intervals with different types: {i1.interval_type} and {i2.interval_type}")

    if hasattr(i1, "__interval_intersection__"):
        return i1.__interval_intersection__(i2)
    elif hasattr(i2, "__interval_intersection__"):
        return i2.__interval_intersection__(i2)

    return _real_interval_intersection(i1, i2)
