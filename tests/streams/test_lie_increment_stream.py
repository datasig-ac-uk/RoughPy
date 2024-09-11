#  Copyright (c) 2023 the RoughPy Developers. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification,
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
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
#  OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
#  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pytest
import numpy as np
import roughpy as rp


from roughpy.streams.lie_increment_stream import LieIncrementStream

def test_lie_increment_stream_metadata():

    stream = LieIncrementStream(np.array([[0., 1., 2.], [3., 4., 5.]]), depth=2) # TODO: improve constructor, stream = LieIncrementStream.from_increments(np.array([[0., 1., 2.],[3., 4., 5.]]), depth=2)

    interval = rp.RealInterval(0.0, 0.1)
    ctx = rp.get_context(width=3, depth=2, coeffs=rp.DPReal)

    assert stream.stream_metadata().width == 0

def test_lie_increment_stream_log_signature_impl():

    stream = LieIncrementStream(np.array([[0., 1., 2.], [3., 4., 5.]]), depth=2) # TODO: improve constructor, stream = LieIncrementStream.from_increments(np.array([[0., 1., 2.],[3., 4., 5.]]), depth=2)

    interval = rp.RealInterval(0.0, 0.1)
    ctx = rp.get_context(width=3, depth=2, coeffs=rp.DPReal)

    expected = rp.Lie([0., 1., 2.], ctx=ctx)
    result = stream.log_signature_impl(interval, ctx)

    assert expected == result

def test_lie_increment_stream_log_signature_empty_false():

    stream = LieIncrementStream(np.array([[0., 1., 2.], [3., 4., 5.]]), depth=2) # TODO: improve constructor, stream = LieIncrementStream.from_increments(np.array([[0., 1., 2.],[3., 4., 5.]]), depth=2)

    interval = rp.RealInterval(0.0, 0.1)

    expected = False
    result = stream.empty(interval)

    assert expected == result

def test_lie_increment_stream_log_signature_empty_true():

    stream = LieIncrementStream()

    interval = rp.RealInterval(0.0, 0.1)

    expected = True
    result = stream.empty(interval)

    assert expected == result


