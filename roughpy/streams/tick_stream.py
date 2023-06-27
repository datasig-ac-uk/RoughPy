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
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Optional, Any, NamedTuple

from roughpy._roughpy import TickStreamConstructionHelper, StreamSchema, ChannelType


class BaseTickDataParser(ABC):
    helper: TickStreamConstructionHelper

    class TickItem(NamedTuple):
        timestamp: Any
        label: str
        type: ChannelType
        data: Any

    def __init__(self, schema: Optional[StreamSchema] = None, schema_only: bool = False):
        if schema is not None:
            self.helper = TickStreamConstructionHelper(schema, schema_only)
        else:
            self.helper = TickStreamConstructionHelper(schema_only)

    @abstractmethod
    def parse_data(self, data: Any):
        pass

    def convert_channel_type(self, ch_type: Any) -> ChannelType:

        if isinstance(ch_type, ChannelType):
            return ch_type

        if not isinstance(ch_type, str):
            raise TypeError(f"cannot convert {ch_type.__name__} to channel type")

        if ch_type == "increment":
            return ChannelType.IncrementChannel

        if ch_type == "value":
            return ChannelType.ValueChannel

        if ch_type == "categorical":
            return ChannelType.CategoricalChannel

    def insert(self, item: TickItem):
        type = self.convert_channel_type(item.type)

        if type == ChannelType.IncrementChannel:
            self.helper.add_increment(item.label, item.timestamp, item.data)
        elif type == ChannelType.ValueChannel:
            self.helper.add_value(item.label, item.timestamp, item.data)
        elif type == ChannelType.CategoricalChannel:
            self.helper.add_categorical(item.label, item.timestamp, item.data)


class StandardTickDataParser(BaseTickDataParser):

    def parse_data(self, data: Any):
        for item in self.visit(data, ["timestamp", "label", "type", "data"], None):
            self.insert(item)

    def visit(self,
              data: Any,
              labels_remaining: list[str],
              current: Optional[dict]
              ):
        yield from getattr(self, f"handle_{type(data).__name__}", self.handle_any)(data, labels_remaining,
                                                                                   current or {})

    def handle_dict(self,
                    data: dict,
                    labels_remaining: list[str],
                    current: dict
                    ):

        if all(label in data for label in labels_remaining):
            yield self.TickItem(**current, **{label: data[label] for label in labels_remaining})
            return

        key_type, *value_types = labels_remaining
        if key_type == "data":
            yield self.TickItem(**current, data=data)
            return

        for key, value in data.items():
            yield from self.visit(value, value_types, {key_type: key, **current})

    def handle_list(self,
                    data: Any,
                    labels_remaining: list[str],
                    current: dict):
        first_type, *remaining = labels_remaining
        if first_type == "data":
            yield self.TickItem(**current, data=data)
            return

        for value in data:
            yield from self.visit(value, labels_remaining, current)

    def handle_tuple(self,
                     data: Any,
                     labels_remaining: list[str],
                     current: dict):
        if len(data) == len(labels_remaining):
            yield self.TickItem(
                **(current or {}), **dict(zip(labels_remaining, data))
            )
            return

        first_label, *other_labels = labels_remaining
        if len(data) == 2:
            yield from self.visit(data[1], other_labels, {**current, first_label: data[0]})
        else:
            yield from self.visit(data[1:], other_labels, {**current, first_label: data[0]})

    def handle_any(self, data, labels_remaining, current):

        if len(labels_remaining) == 1:
            yield self.TickItem(**current, data=data)
            return

        if "label" in labels_remaining or "timestamp" in labels_remaining:
            raise ValueError("cannot infer timestamp or label from a single data value")

        if isinstance(data, (float, int)):
            # infer value type
            yield self.TickItem(**current, type="increment", data=data)
        elif isinstance(data, str):
            # infer categorical
            yield self.TickItem(**current, type="categorical", data=data)
        else:
            raise ValueError("other types cannot be used for anything but data value")
