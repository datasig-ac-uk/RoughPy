import enum
import math
import typing

from dataclasses import dataclass
from typing import Protocol, TypeVar, TypeAlias, Self, Any, Generic


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
    def interval_type(self) -> IntervalType:
        ...

    @property
    def inf(self) -> float:
        ...

    @property
    def sup(self) -> float:
        ...




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
    def inf(self) -> float:
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return math.ldexp(self.k, -self.n)

    @property
    def sup(self) -> float:
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return math.ldexp(self.k+1, -self.n)


@dataclass(frozen=True)
class RealInterval(Generic[RealT]):
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
    _inf : RealT
    _sup : RealT
    _interval_type: IntervalType

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> float:
        return self._inf

    @property
    def sup(self) -> float:
        return self._sup



@dataclass(frozen=True)
class Partition(Generic[RealT]):
    _endpoints: list[RealT]
    _interval_type: IntervalType

    def __len__(self) -> int:
        return len(self._endpoints) - 1

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> float:
        return float(self._endpoints[0])

    @property
    def sup(self) -> float:
        return float(self._endpoints[-1])