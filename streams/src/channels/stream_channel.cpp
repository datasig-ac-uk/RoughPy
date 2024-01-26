// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by sam on 07/08/23.
//


#include <roughpy/streams/channels/stream_channel.h>

#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>


using namespace rpy;
using namespace rpy::streams;

StreamChannel::~StreamChannel() {}

dimn_t StreamChannel::num_variants() const
{
    return 1;
}

string StreamChannel::label_suffix(rpy::dimn_t variant_no) const
{
    return "";
}

dimn_t StreamChannel::variant_id_of_label(string_view label) const { return 0; }
void StreamChannel::set_lie_info(
        deg_t width, deg_t depth, algebra::VectorType vtype
)
{}
StreamChannel& StreamChannel::add_variant(string variant_label)
{
    return *this;
}
StreamChannel& StreamChannel::insert_variant(string variant_label)
{
    return *this;
}
const std::vector<string>& StreamChannel::get_variants() const
{
    static const std::vector<string> no_variants;
    return no_variants;
}

void StreamChannel::set_lead_lag(bool new_value) {}
bool StreamChannel::is_lead_lag() const { return false; }
void StreamChannel::convert_input(
        scalars::ScalarArray& dst, const scalars::ScalarArray& src
) const
{
    if (src.empty() == 0) { return; }
    RPY_CHECK(!src.is_null());
    RPY_CHECK(!dst.empty() || dst.type() != nullptr);

    if (dst.empty()) {
        dst = (*dst.type())->allocate(src.size());
    }

    (*dst.type())->convert_copy(dst, src);
}

#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamChannel
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
