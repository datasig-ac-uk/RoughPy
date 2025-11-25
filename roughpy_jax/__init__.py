import ctypes

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import jax

from roughpy_jax.algebra import (
    LieBasis, TensorBasis, DenseFreeTensor, DenseShuffleTensor, DenseLie, ft_fma, ft_mul, ft_exp, ft_log, ft_fmexp,
    st_fma, st_mul, antipode)

from roughpy_jax.intervals import (Interval, IntervalType, RealInterval, DyadicInterval, Partition)

from roughpy_jax.streams import (Stream, ValueStream, PiecewiseAbelianStream, DyadicCachedTickStream,
                                 TensorValuedStream)
