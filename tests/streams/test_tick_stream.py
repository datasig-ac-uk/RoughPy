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

from roughpy import DPReal, Lie, RealInterval, TickStream

DATA_FORMATS = [
    {
        1.0: [
            ("first", "increment", 1.0),
            ("second", "increment", 1.0)
        ]
    },
    {
        1.0: {
            "first": ("increment", 1.0),
            "second": ("increment", 1.0)
        }
    },
    {
        1.0: [
            {"label": "first", "type": "increment", "data": 1.0},
            {"label": "second", "type": "increment", "data": 1.0},
        ]
    },
    {
        1.0: {
            "first": {"type": "increment", "data": 1.0},
            "second": {"type": "increment", "data": 1.0},
        }
    },
    [
        (1.0, "first", "increment", 1.0),
        (1.0, "second", "increment", 1.0),
    ],
    [
        (1.0, ("first", "increment", 1.0)),
        (1.0, ("second", "increment", 1.0)),
    ],
    [
        (1.0, {"label": "first", "type": "increment", "data": 1.0}),
        (1.0, {"label": "second", "type": "increment", "data": 1.0}),
    ],
    [
        (1.0, [
            ("first", "increment", 1.0),
            ("second", "increment", 1.0),
        ])
    ],
    [
        (1.0, [
            {"label": "first", "type": "increment", "data": 1.0},
            {"label": "second", "type": "increment", "data": 1.0},
        ])
    ],
    [
        (1.0, {
            "first": ("increment", 1.0),
            "second": ("increment", 1.0),
        })
    ],
    {
        1.0: [
            ("first", "increment", 1.0),
            {
                "label": "second",
                "type": "increment",
                "data": 1.0
            }
        ]
    },
    {
        1.0: [
            ("first", "increment", 1.0),
            {
                "second": ("increment", 1.0)
            }
        ]
    },
    {
        1.0: [
            ("first", "increment", 1.0),
            {
                "second": {
                    "type": "increment",
                    "data": 1.0
                }
            }
        ]
    },
    {
        1.0: {
            "first": ("increment", 1.0),
            "second": {
                "type": "increment",
                "data": 1.0
            }
        }
    }

]


@pytest.mark.parametrize("data", DATA_FORMATS)
def test_construct_tick_path_from_data(data):
    stream = TickStream.from_data(data, width=2, depth=2, dtype=DPReal)

    assert stream.width == 2
    lsig = stream.log_signature(RealInterval(0.0, 2.0), 2)

    expected = Lie([1.0, 1.0, 0.5], width=2, depth=2, dtype=DPReal)
    assert lsig == expected, f"{lsig} == {expected}"




def test_construct_tick_stream_with_time(rng):
    data = DATA_FORMATS[0]

    stream = TickStream.from_data(data, width=2, depth=2, include_time=True,
                                  dtype=DPReal)

    assert stream.width == 3
