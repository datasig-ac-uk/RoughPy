#  Copyright (c) 2023 the RoughPy Developers. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification,
#  are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its contributors
#  may be used to endorse or promote products derived from this software without
#  specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import typing

from typing import Protocol, Self, Callable

import numpy as np

from roughpy.typing import LieT, GroupT, ParamT, BasisT, StreamValueT, PartitionT
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


## Raw data streams
# These streams are think masks around raw data. These provide stream-like
# access to raw data but might not meet the complexity requirements for real
# work. These are the entry point to rough path based learning pipelines but,
# internally, these should be converted to intermediate streams of various kinds
# to facilitate access with the correct complexity guarantees.


class TickStream: ...


class TimeSeries: ...


class LieIncrementStream: ...


## Intermediate streams
# These streams are particular views of streams. These offer better complexity
# guarantees than raw data streams but have a more complicated internal
# representation and query mechanism. These should be used internally when many
# queries are expected in the learning process. These streams often aggregate
# data in a way that typically makes recovering the underlying data tricky. For
# instance, the stream may aggregate large numbers of increments over a partition
# or according to dyadic dissection of the support. These may also internally
# reparametrise the streams, though the external interface will remain consistent.


class PiecewiseAbelianStream: ...


class DyadicLieIncrementStream: ...


class DynamicDyadicCachedStream: ...


class RestrictedStream: ...


## Advanced streams


class TransformedStream: ...


def simplify(stream: Stream, partition: PartitionT) -> PiecewiseAbelianStream: ...


def make_cached_stream(
    stream: Stream, resolution: int = None, dtype: np.dtype = None
) -> DyadicLieIncrementStream: ...
