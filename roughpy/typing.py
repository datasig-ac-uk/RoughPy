import typing

from typing import TypeVar


# Type alias for stream and interval parameterisation
ParamT = TypeVar("ParamT")

PartitionT = TypeVar("PartitionT")

# Type for basis objects in generic algebra routines
BasisT = TypeVar("BasisT")

# The type of the "signature" of a stream developed into a Lie group
GroupT = TypeVar("GroupT")

# The type of the "log_signature" of a stream developed into a Lie group
LieT = TypeVar("LieT")

# The value type associated with a value stream
StreamValueT = TypeVar("StreamValueT")



class Interval(typing.Generic[ParamT]):
    ...


