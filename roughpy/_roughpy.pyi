import enum

from abc import abstractmethod, ABC
from typing import (
    Any,
    Union,
    overload,
    Sequence,
    Dict,
    Final,
    List
)


class ScalarTypeMeta(type):
    ...


class ScalarTypeBase(metaclass=ScalarTypeMeta):
    ...


class SPReal(ScalarTypeBase):
    ...


class DPReal(ScalarTypeBase):
    ...


class Rational(ScalarTypeBase):
    ...

class HPReal(ScalarTypeBase): ...
class BFloat16(ScalarTypeBase): ...


class Scalar:

    def scalar_type(self) -> ScalarTypeMeta:
        ...

    def __neg__(self) -> Scalar: ...

    def __add__(self, other: Scalar) -> Scalar: ...

    def __sub__(self, other: Scalar) -> Scalar: ...

    def __mul__(self, other: Scalar) -> Scalar: ...

    def __div__(self, other: Scalar) -> Scalar: ...

    def __iadd__(self, other: Scalar) -> Scalar: ...

    def __isub__(self, other: Scalar) -> Scalar: ...

    def __imul__(self, other: Scalar) -> Scalar: ...

    def __idiv__(self, other: Scalar) -> Scalar: ...

    def __eq__(self, other: Scalar) -> bool: ...

    def __ne__(self, other: Scalar) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


ScalarLike = Union[Scalar, float, int]


class IntervalType(enum.Enum):
    Clopen = ...
    # Opencl = ...   # Currently disabled until support is added in the underlying library


class Interval(ABC):
    interval_type: IntervalType = ...

    @abstractmethod
    def inf(self) -> float: ...

    @abstractmethod
    def sup(self) -> float: ...

    def included_end(self) -> float: ...

    def excluded_end(self) -> float: ...

    def intersects_with(self, other: Interval) -> bool: ...

    def contains(self, other: Union[float, Interval]) -> bool: ...

    def __eq__(self, other: Interval) -> bool: ...

    def __ne__(self, other: Interval) -> bool: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...


class RealInterval(Interval):

    def inf(self) -> float: ...

    def sup(self) -> float: ...


class Dyadic:
    k: Final[int] = ...
    n: Final[int] = ...

    @staticmethod
    def dyadic_equals(lhs: Dyadic, rhs: Dyadic) -> bool: ...

    @staticmethod
    def rational_equals(lhs: Dyadic, rhs: Dyadic) -> bool: ...

    def rebase(self, resolution: int): ...

    def __float__(self) -> float: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __lt__(self, other: Dyadic) -> bool: ...

    def __le__(self, other: Dyadic) -> bool: ...

    def __gt__(self, other: Dyadic) -> bool: ...

    def __ge__(self, other: Dyadic) -> bool: ...

    def __iadd__(self, other: int): ...


class DyadicInterval(Interval, Dyadic):
    def inf(self) -> float: ...

    def sup(self) -> float: ...

    def dyadic_included_end(self) -> Dyadic: ...

    def dyadic_excluded_end(self) -> Dyadic: ...

    def dyadic_inf(self) -> Dyadic: ...

    def dyadic_sup(self) -> Dyadic: ...

    def shrink_to_contained_end(self, arg: int = ...): ...

    def shrink_to_omitted_end(self): ...

    def shrink_left(self): ...

    def shrink_right(self): ...

    @staticmethod
    def to_dyadic_intervals(interval: Interval,
                            resolution: int,
                            interval_type: IntervalType) -> Sequence[Interval]: ...

    @overload
    @staticmethod
    def to_dyadic_intervals(inf: float,
                            sup: float,
                            resolution: int,
                            interval_type: IntervalType) -> Sequence[Interval]: ...


class VectorType(enum.Enum):
    DenseVector = ...
    SparseVector = ...


class TensorKey:
    width: Final[int] = ...
    max_degree: Final[int] = ...

    def degree(self) -> int: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: TensorKey): ...


class LieKey:
    def __str__(self) -> str: ...


class TensorKeyIterator:

    def __next__(self) -> TensorKey: ...


class LieKeyIterator:

    def __next__(self) -> LieKey: ...


class Context:
    ...


def get_context(width: int,
                depth: int,
                coeffs: Union[None, str, type, ScalarTypeMeta] = ...
                ) -> Context: ...


class LieIteratorItem:

    def key(self) -> LieKey: ...

    def value(self) -> Scalar: ...


class FreeTensorIteratorItem:

    def key(self) -> TensorKey: ...

    def value(self) -> Scalar: ...


class ShuffleTensorIteratorItem:

    def key(self) -> TensorKey: ...

    def value(self) -> Scalar: ...


class Lie:
    width: Final[int] = ...
    max_degree: Final[int] = ...
    dtype: Final[ScalarTypeMeta] = ...
    storage_type: Final[VectorType] = ...

    def size(self) -> int: ...

    def dimension(self) -> int: ...

    def degree(self) -> int: ...

    def __iter__(self): ...

    def __neg__(self) -> Lie: ...

    def __add__(self, other: Lie) -> Lie: ...

    def __sub__(self, other: Lie) -> Lie: ...

    def __mul__(self, other: ScalarLike) -> Lie: ...

    def __rmul__(self, other: ScalarLike) -> Lie: ...

    def __truediv__(self, other: ScalarLike) -> Lie: ...

    @overload
    def __mul__(self, other: Lie) -> Lie: ...

    def __iadd__(self, other: Lie): ...

    def __isub__(self, other: Lie): ...

    def __imul__(self, other: ScalarLike): ...

    def __itruediv__(self, other: ScalarLike): ...

    @overload
    def __imul__(self, other: Lie): ...

    def add_scal_mul(self, other: Lie, scalar: ScalarLike): ...

    def sub_scal_mul(self, other: Lie, scalar: ScalarLike): ...

    def add_scal_div(self, other: Lie, scalar: ScalarLike): ...

    def sub_scal_div(self, other: Lie, scalar: ScalarLike): ...

    def add_mul(self, lhs: Lie, rhs: Lie): ...

    def sub_mul(self, lhs: Lie, rhs: Lie): ...

    def mul_smul(self, other: Lie, scalar: ScalarLike): ...

    def mul_sdiv(self, other: Lie, scalar: ScalarLike): ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: Lie) -> bool: ...

    def __ne__(self, other: Lie) -> bool: ...

    def __getitem__(self, item: LieKey) -> Scalar: ...


class FreeTensor:
    width: Final[int] = ...
    max_degree: Final[int] = ...
    dtype: Final[ScalarTypeMeta] = ...
    storage_type: Final[VectorType] = ...

    def size(self) -> int: ...

    def dimension(self) -> int: ...

    def degree(self) -> int: ...

    def __iter__(self): ...

    def __neg__(self) -> FreeTensor: ...

    def __add__(self, other: FreeTensor) -> FreeTensor: ...

    def __sub__(self, other: FreeTensor) -> FreeTensor: ...

    def __mul__(self, other: ScalarLike) -> FreeTensor: ...

    def __rmul__(self, other: ScalarLike) -> FreeTensor: ...

    def __truediv__(self, other: ScalarLike) -> FreeTensor: ...

    @overload
    def __mul__(self, other: FreeTensor) -> FreeTensor: ...

    def __iadd__(self, other: FreeTensor): ...

    def __isub__(self, other: FreeTensor): ...

    def __imul__(self, other: ScalarLike): ...

    def __itruediv__(self, other: ScalarLike): ...

    @overload
    def __imul__(self, other: FreeTensor): ...

    def add_scal_mul(self, other: FreeTensor, scalar: ScalarLike): ...

    def sub_scal_mul(self, other: FreeTensor, scalar: ScalarLike): ...

    def add_scal_div(self, other: FreeTensor, scalar: ScalarLike): ...

    def sub_scal_div(self, other: FreeTensor, scalar: ScalarLike): ...

    def add_mul(self, lhs: FreeTensor, rhs: FreeTensor): ...

    def sub_mul(self, lhs: FreeTensor, rhs: FreeTensor): ...

    def mul_smul(self, other: FreeTensor, scalar: ScalarLike): ...

    def mul_sdiv(self, other: FreeTensor, scalar: ScalarLike): ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: FreeTensor) -> bool: ...

    def __ne__(self, other: FreeTensor) -> bool: ...

    def __getitem__(self, item: TensorKey) -> Scalar: ...


class ShuffleTensor:
    width: Final[int] = ...
    max_degree: Final[int] = ...
    dtype: Final[ScalarTypeMeta] = ...
    storage_type: Final[VectorType] = ...

    def size(self) -> int: ...

    def dimension(self) -> int: ...

    def degree(self) -> int: ...

    def __iter__(self): ...

    def __neg__(self) -> ShuffleTensor: ...

    def __add__(self, other: ShuffleTensor) -> ShuffleTensor: ...

    def __sub__(self, other: ShuffleTensor) -> ShuffleTensor: ...

    def __mul__(self, other: ScalarLike) -> ShuffleTensor: ...

    def __rmul__(self, other: ScalarLike) -> ShuffleTensor: ...

    def __truediv__(self, other: ScalarLike) -> ShuffleTensor: ...

    @overload
    def __mul__(self, other: ShuffleTensor) -> ShuffleTensor: ...

    def __iadd__(self, other: ShuffleTensor): ...

    def __isub__(self, other: ShuffleTensor): ...

    def __imul__(self, other: ScalarLike): ...

    def __itruediv__(self, other: ScalarLike): ...

    @overload
    def __imul__(self, other: ShuffleTensor): ...

    def add_scal_mul(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def sub_scal_mul(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def add_scal_div(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def sub_scal_div(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def add_mul(self, lhs: ShuffleTensor, rhs: ShuffleTensor): ...

    def sub_mul(self, lhs: ShuffleTensor, rhs: ShuffleTensor): ...

    def mul_smul(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def mul_sdiv(self, other: ShuffleTensor, scalar: ScalarLike): ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __eq__(self, other: ShuffleTensor) -> bool: ...

    def __ne__(self, other: ShuffleTensor) -> bool: ...

    def __getitem__(self, item: TensorKey) -> Scalar: ...


class Stream: ...


class LieIncrementStream: ...


class FunctionStream: ...


class PiecewiseAbelianStream: ...


class ExternalDataStream: ...


class BrownianStream: ...


class TickStream:
    @staticmethod
    def from_data(data: Any, **kwargs): ...


class StreamSchema:

    @staticmethod
    def from_data(data: Union[Dict[float, Union[str, float, int]], Sequence[tuple]]) -> StreamSchema:
        ...

    @staticmethod
    def parse(schema: List[Dict[str, Union[str, List[str]]]]) -> StreamSchema: ...

    def get_labels(self) -> list[str]: ...


class TickStreamConstructionHelper:

    def __init__(self): ...

    @overload
    def __init__(self, schema_only: bool): ...

    @overload
    def __init__(self, schema: StreamSchema, schema_only: bool): ...
