#  Copyright (c) 2023 Datasig Developers. All rights reserved.
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


import pytest

from roughpy import StreamSchema



@pytest.fixture
def sample_data_dict():
    return {
        1.0: ("first", 1.0),
        2.0: ("first", 2.0),
        3.0: ("second", "cat1"),
        4.0: ("second", "cat2"),
        5.0: ("third", "value", 1.0),
    }

@pytest.fixture
def sample_data_seq(sample_data_dict):
    return [(ts, *args) for ts, args in sample_data_dict.items()]

def test_schema_from_dict(sample_data_dict):

    schema = StreamSchema.from_data(sample_data_dict)

    assert schema.get_labels() == [
        "first",
        "second:cat1",
        "second:cat2",
        "third:lead",
        "third:lag"
    ]


def test_schema_from_seq(sample_data_seq):

    schema = StreamSchema.from_data(sample_data_seq)

    assert schema.get_labels() == [
        "first",
        "second:cat1",
        "second:cat2",
        "third:lead",
        "third:lag"
    ]

@pytest.fixture
def json_like_schema():
    return [
        {
            "label": "first",
            "type": "increment"
        },
        {
            "label": "second",
            "type": "value",
        },
        {
            "label": "third",
            "type": "categorical",
            "variants": ["cat1", "cat2"]
        },
        {
            # No label, "3"
            "type": "increment"
        },
        {
            # No label or type, derived categorical, "4"
            "variants": ["cata", "catb"]
        },
        {
            # No label, "5"
            "type": "value"
        }
    ]


def test_parse_jsonlike(json_like_schema):
    schema = StreamSchema.parse(json_like_schema)

    assert schema.get_labels() == [
        "first",
        "second:lead",
        "second:lag",
        "third:cat1",
        "third:cat2",
        "3",
        "4:cata",
        "4:catb",
        "5:lead",
        "5:lag"
    ]
