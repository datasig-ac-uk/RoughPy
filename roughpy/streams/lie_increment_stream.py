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

from roughpy._roughpy import StreamInterface, VectorType, IntervalType


class LieIncrementStream(StreamInterface):

    def __init__(self):

        StreamInterface.__init__(self)

        # args m_data, schema ?

        # 6 methods, start with 2: log_signature_impl, empty. Try to avoid using ctx due to current design changes.

    def log_signature_impl(self, interval, ctx):
        # See lines 159>184 of RoughPy/streams/src/lie_increment_stream.cpp

        # TODO: implement m_data, a list of pairs, the stream itself (lie increment stream)

        begin = self.m_data.lower_bound(interval.inf())
        end = self.m_data.lower_bound(interval.sup())

        if begin == end:
            return ctx.zero_lie(self.m_data.cached_vector_type)

        lies = []  # create a list of lies

        # Iterate over the Lies where the parameter lies between inf and sup
        for i in range(0):  # length of the stream

            # check parameter lies between inf and sup

            param = None  # TODO: implement param
            lie = None # TODO: implement lie
            if begin < param < end:  # check if this elt param between inf and sup
                # if true, append to list of Lies
                lies.append(lie)

        return ctx.cbh(lies, VectorType.DenseVector)

    def empty(self, interval):
        # See lines 185>197 of RoughPy/streams/src/lie_increment_stream.cpp

        # TODO: define m_data, intervals::IntervalType::Opencl, interval

        if interval.type == IntervalType.Opencl:
            begin = self.m_data.upper_bound(interval.inf())
            end = self.m_data.upper_bound(interval.sup())
        else:
            begin = self.m_data.lower_bound(interval.inf())
            end = self.m_data.lower_bound(interval.sup())

        # Iterate through the whole list
        for i in range(0):  # length of the stream
            # Check if there's anything between inf and sup. If you find anything, then false.
            param = None  # TODO, implement param
            if begin < param < end:  # check if this elt param between inf and sup
                return False
        return True
