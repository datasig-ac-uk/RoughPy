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

import roughpy as rp
from roughpy import DPReal, Lie, RealInterval, TickStream

DATA_FORMATS = [
    {1.0: [("first", "increment", 1.0), ("second", "increment", 1.0)]},
    {1.0: {"first": ("increment", 1.0), "second": ("increment", 1.0)}},
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
        (
            1.0,
            [
                ("first", "increment", 1.0),
                ("second", "increment", 1.0),
            ],
        )
    ],
    [
        (
            1.0,
            [
                {"label": "first", "type": "increment", "data": 1.0},
                {"label": "second", "type": "increment", "data": 1.0},
            ],
        )
    ],
    [
        (
            1.0,
            {
                "first": ("increment", 1.0),
                "second": ("increment", 1.0),
            },
        )
    ],
    {
        1.0: [
            ("first", "increment", 1.0),
            {"label": "second", "type": "increment", "data": 1.0},
        ]
    },
    {1.0: [("first", "increment", 1.0), {"second": ("increment", 1.0)}]},
    {
        1.0: [
            ("first", "increment", 1.0),
            {"second": {"type": "increment", "data": 1.0}},
        ]
    },
    {1.0: {"first": ("increment", 1.0), "second": {"type": "increment", "data": 1.0}}},
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

    stream = TickStream.from_data(
        data, width=2, depth=2, include_time=True, dtype=DPReal
    )

    assert stream.width == 3


def test_tick_stream_float_timestamps_are_absolute():
    """
    Regression test for issue where float timestamps were being shifted relative
    to the first data point. They should be treated as absolute parameters (reference 0.0).
    """
    ctx = rp.get_context(3, 2, DPReal)

    # Simple schema
    schema = [
        ("time", "value", {"lead_lag": False}),
        ("s1", "value", {"lead_lag": True}),
    ]

    # Data starting at t=3.0
    # If timestamps were relative to first point, this would look like starting at t=0.0
    raw_data = [
        (3.0, "s1", "value", 3.0),
        (5.0, "s1", "value", 3.0),
    ]

    data_with_time = []
    for t, label, typ, val in raw_data:
        data_with_time.append((t, "time", "value", t))
        data_with_time.append((t, label, typ, val))

    support = RealInterval(0, 7)

    s1 = TickStream.from_data(
        data_with_time,
        schema=schema,
        ctx=ctx,
        support=support,
    )

    # Query interval [0, 2). Since data starts at 3.0, this should be empty/trivial.
    interval_empty = RealInterval(0.0, 2.0)
    sig_empty = s1.signature(interval_empty, ctx=ctx, resolution=10)

    # Should be identity (1.0)
    # Checking if it has only one component (scalar) and it is 1.0
    # Or just check size if it's sparse?
    # But FreeTensor might not be sparse in python binding access

    # We can check specific coefficients
    # The scalar component (empty word) should be 1.0
    # All other components should be 0.0

    # In RoughPy, converting to list/dict helps.
    # sig_empty should be effectively 1.

    # Using str(sig_empty) to check
    assert str(sig_empty) == "{ 1() }"

    # Query interval [0, 4) which contains the first jump at 3.0.
    interval_first = RealInterval(0.0, 4.0)
    sig_first = s1.signature(interval_first, ctx=ctx, resolution=10)

    # Should NOT be identity
    assert str(sig_first) != "{ 1() }"

    # Check that it captured the jump
    # The jump is (3, 3, 3) at t=3.0.
    # We can check that the "time" channel (1) increment is 3.0
    # Note: accessing coefficients might need specific API.
    # But string representation check is robust enough for regression here.
