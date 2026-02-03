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


def test_tick_stream_lead_lag():
    import roughpy as rp

    # Data from the bug report
    data = [
        (0.0, "s1", "value", 1.0),
        (1.0, "s1", "value", 2.0),
        (2.0, "s1", "value", 3.0),
        (3.0, "s1", "value", 2.0),
    ]

    # Manual implementation (known correct)
    ll_data = []
    for i in range(len(data)):
        t_curr, _, _, v_curr = data[i]

        if i == 0:
            ll_data.append((t_curr, "lead", "value", v_curr))
            ll_data.append((t_curr, "lag", "value", v_curr))
        else:
            t_prev, _, _, v_prev = data[i - 1]

            ll_data.append((t_curr, "lead", "value", v_curr))
            ll_data.append((t_curr, "lag", "value", v_prev))

            ll_data.append((t_curr, "lead", "value", v_curr))
            ll_data.append((t_curr, "lag", "value", v_curr))

    ctx = rp.get_context(2, 2, rp.DPReal)
    schema_manual = [("lead", "value", {}), ("lag", "value", {})]

    s_manual = rp.TickStream.from_data(
        ll_data,
        schema=schema_manual,
        ctx=ctx,
        support=rp.RealInterval(0, 4),
    )

    sig_manual = s_manual.signature(
        rp.RealInterval(1.0, 3.0001), ctx=ctx, resolution=10
    )

    # Native implementation
    s_native = rp.TickStream.from_data(
        data,
        schema=[("s1", "value", {"lead_lag": True})],
        ctx=ctx,
        support=rp.RealInterval(0, 4),
    )
    sig_native = s_native.signature(
        rp.RealInterval(1.0, 3.0001), ctx=ctx, resolution=10
    )

    # They should match
    assert str(sig_native) == str(sig_manual)
