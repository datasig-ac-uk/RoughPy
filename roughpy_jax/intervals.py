import enum
import math
import numbers
import typing
from typing import Protocol, TypeVar, TypeAlias, Self, Any, Generic
from dataclasses import dataclass

import jax.numpy as jnp
import jax


RealT = TypeVar("RealT")

class IntervalType(enum.IntEnum):
    ClOpen = 0
    OpenCl = 1

def _interval_dataclass(cls):
    """
    Combined decorator for roughpy_jax interval objects

    Registers dataclass and JAX data class with dynamic inf and sup and static 
    interval_type
    """
    cls = dataclass(cls, frozen=True)
    return jax.tree_util.register_dataclass(
        cls, data_fields=["_inf", "_sup"], meta_fields=["_interval_type"]
    )


def _partition_dataclass(cls):
    """
    Combined decorator for roughpy_jax partition objects

    Registers dataclass and JAX data class with dynamic endpoints and 
    interval_type
    """
    cls = dataclass(cls, frozen=True)
    return jax.tree_util.register_dataclass(
        cls, data_fields=["_endpoints"], meta_fields=["_interval_type"]
    )

@typing.runtime_checkable
class Interval[RealT](Protocol):
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
    def inf(self) -> RealT:
        ...

    @property
    def sup(self) -> RealT:
        ...
    
    @property
    def length(self) -> RealT:
        ...
        
    def intersection(self, other: Self) -> typing.Optional[Self]:
        """
        Calculate the intersection of this interval with another interval.
        :param other: The other interval to intersect with.
        :type other: Interval
        :return: A new Interval representing the intersection, or None if there is no intersection.
        :rtype: typing.Optional[Interval]
        """
        ...


class BaseInterval(Interval):
    # TODO: These don't need to be in a class, just have module-level functions for str, length, and intersection that 
    # take Intervals as arguments. The only reason to have these in a class is if we want to use inheritance to share 
    # code between different Interval implementations.
    
    @staticmethod
    def __str__(interval) -> str:
        reprs = {
            IntervalType.ClOpen: "[{}, {})",
            IntervalType.OpenCl: "({}, {}]",
        }
        return reprs[interval.interval_type].format(interval.inf, interval.sup)
    
    @staticmethod
    def length(interval: Interval) -> RealT:
        """
        Calculate the length of the interval.
        :return: The length of the interval, calculated as sup - inf.
        :rtype: float
        """
        return jnp.maximum(0.0, interval.sup - interval.inf)
    
    @staticmethod
    def intersection(
            left_interval: Interval, 
            right_interval: Interval,
        ) -> typing.Optional[Interval]:
        """
        Calculate the intersection of this interval with another interval.
        :param other: The other interval to intersect with.
        :type other: Interval
        :return: A new Interval representing the intersection, or None if there is no intersection.
        :rtype: typing.Optional[RealInterval]
        """
        if not isinstance(left_interval, Interval) or not isinstance(right_interval, Interval):
            raise TypeError("Both arguments must be of type Interval")
        
        if left_interval.interval_type != right_interval.interval_type:
            raise TypeError("Both intervals must be of the same IntervalType")
        
        new_inf = jnp.maximum(left_interval.inf, right_interval.inf)
        new_sup = jnp.minimum(left_interval.sup, right_interval.sup)

        # NOTE: If we just return a 0-length interval when there is no intersection, 
        # then we can avoid returning None which is not jittable, and instead 
        # use a jax.lax.cond to return the identity tensor on a zeroed 
        # intersection length.
        # if new_inf >= new_sup:
        #     return None  # No intersection
        
        IntervalType = left_interval.__class__
        interval_type = left_interval.interval_type

        # Intersections are defined by bounds, but not every Interval implementation
        # can be constructed from (inf, sup, interval_type) (e.g. DyadicInterval).
        # Use a canonical RealInterval result to avoid incorrect reconstruction.
        # NOTE: Do NOT call float() here — new_inf/new_sup may be JAX tracers during JIT.
        return IntervalType(_inf=new_inf, _sup=new_sup, _interval_type=interval_type)


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
            raise TypeError(f"Dyadic.k must be an integer-like value, got {type(k).__name__}: {k!r}")

        if isinstance(n, numbers.Integral):
            n_int = int(n)
        elif isinstance(n, float) and n.is_integer():
            n_int = int(n)
        else:
            raise TypeError(f"Dyadic.n must be an integer-like value, got {type(n).__name__}: {n!r}")

        if n_int < 0:
            raise ValueError(f"Dyadic.n must be non-negative, got {n_int}")

        object.__setattr__(self, "k", k_int)
        object.__setattr__(self, "n", n_int)

    def __float__(self) -> float:
        return math.ldexp(self.k, -self.n)

    def __jax_array__(self):
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
    def inf(self) -> float:
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return math.ldexp(self.k, -self.n)

    @property
    def sup(self) -> float:
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return math.ldexp(self.k+1, -self.n)
    
    @property
    def length(self) -> float:
        return BaseInterval.length(self)
    
    def intersection(self, other: Self) -> typing.Optional[Interval]:
        """
        Calculate the intersection of this dyadic interval with another dyadic interval.
        :param other: The other dyadic interval to intersect with.
        :type other: DyadicInterval
        :return: A new DyadicInterval representing the intersection, or None if there is no intersection.
        :rtype: typing.Optional[DyadicInterval]
        """
        # TODO: How to handle intersection of DyadicInterval with non-DyadicInterval? 
        # Should we return a RealInterval instead? For now we just require both 
        # to be DyadicInterval for intersection.
        if not isinstance(other, DyadicInterval):
            raise TypeError("A DyadicInterval can only be intersected with another DyadicInterval")
        raise NotImplementedError("Intersection of DyadicIntervals is not yet implemented")


@_interval_dataclass
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

    def __str__(self) -> str:
        return BaseInterval.__str__(self)
    
    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> RealT:
        return self._inf

    @property
    def sup(self) -> RealT:
        return self._sup
    
    @property
    def length(self) -> RealT:
        return BaseInterval.length(self)
    
    def intersection(self, other: Self) -> typing.Optional[Self]:
        """
        Calculate the intersection of this real interval with another real interval.
        :param other: The other real interval to intersect with.
        :type other: RealInterval
        :return: A new RealInterval representing the intersection, or None if there is no intersection.
        :rtype: typing.Optional[RealInterval]
        """
        return BaseInterval.intersection(self, other)


@_partition_dataclass
class Partition(Generic[RealT]):
    _endpoints: list[RealT]
    _interval_type: IntervalType

    def __len__(self) -> int:
        return len(self._endpoints) - 1
    
    def __str__(self) -> str:
        return BaseInterval.__str__(self)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> RealT:
        return jnp.asarray(self._endpoints[0])

    @property
    def sup(self) -> RealT:
        return jnp.asarray(self._endpoints[-1])
    
    @property
    def length(self) -> RealT:
        return BaseInterval.length(self)
    
    def to_real_interval(self) -> RealInterval[RealT]:
        """
        Convert the partition to a RealInterval.
        :return: A RealInterval representing the partition.
        :rtype: RealInterval
        """ 
        return RealInterval[RealT](
            _inf=self.inf,
            _sup=self.sup,
            _interval_type=self._interval_type,
        )
        
    
    def intersection(self, other: Interval) -> typing.Optional[Self]:
        """
        Calculate the intersection of this partition with another Interval.
        :param other: The other interval to intersect with.
        :type other: Partition
        :return: A new Partition representing the intersection, or None if there is no intersection.
        :rtype: typing.Optional[Partition]
        """
        # (2026-02-16) JL: I think this is the correct implementation of an 
        # intersection between a Partition and an Interval, but it may be 
        # redundant as we need both the original partition and the query 
        # interval to compute the scaling values for the log signature over the 
        # query interval.
                
        # Here we convert to RealInterval to perform the intersection logic
        intermediate_itvl = self.to_real_interval()
        intersect_itvl = intermediate_itvl.intersection(other)
        if intersect_itvl is None:
            return None

        new_endpoints = []
        # 1) Add new inf if it is within bounds of old interval
        if self.inf < intersect_itvl.inf:
            new_endpoints.append(intersect_itvl.inf)
        # 2) Include all inner points
        for ep in self._endpoints:
            if intersect_itvl.inf <= ep <= intersect_itvl.sup:
                new_endpoints.append(ep)
        # 3) Add new sup if within bounds of old interval
        if intersect_itvl.sup < self.sup:
            new_endpoints.append(intersect_itvl.sup)
            
        return Partition[RealT](
            _endpoints=new_endpoints,
            _interval_type=self.interval_type,
        )
        
    def to_intervals(self) -> list[Interval[RealT]]:
        """
        Convert the partition into a list of RealIntervals corresponding to 
        the subintervals defined by the partition.
        :param partition: The Partition to convert.
        :type partition: Partition
        :return: A list of RealIntervals representing the subintervals of the partition.
        :rtype: list[RealInterval]
        """
        # NOTE: typing here should be more generic than RealInterval[float], but this is the only type of interval we
        return [RealInterval[RealT](
            _inf=self._endpoints[i],
            _sup=self._endpoints[i+1],
            _interval_type=self.interval_type,
        ) for i in range(len(self._endpoints)-1)]