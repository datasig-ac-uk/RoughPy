
import typing

from typing import Protocol, Self, Callable

from roughpy.typing import LieT, GroupT, ParamT, BasisT, StreamValueT
from roughpy.typing import Interval


@typing.runtime_checkable
class Stream(Protocol[LieT, GroupT, ParamT]):

    @property
    def lie_basis(self) -> BasisT: ...

    @property
    def group_basis(self) -> BasisT: ...

    @property
    def support(self) -> Interval[ParamT]: ...

    def signature(self, interval: Interval[ParamT]) -> GroupT: ...

    def log_signature(self, interval: Interval[ParamT]) -> LieT: ...

    def restrict(self, interval: Interval[ParamT]) -> Stream[LieT, GroupT, ParamT]: ...


@typing.runtime_checkable
class ValueStream(Protocol[StreamValueT, LieT, GroupT, ParamT]):

    @property
    def stream(self) -> Stream[LieT, GroupT, ParamT]: ...

    @property
    def value_propagate_function(
            self,
    ) -> Callable[[Interval[ParamT], StreamValueT], StreamValueT]: ...

    @property
    def base_value(self) -> StreamValueT: ...

    def query(self, interval: Interval[ParamT]) -> Self: ...

