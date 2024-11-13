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

#include <roughpy/streams/channels/categorical_channel.h>

#include "roughpy/core/check.h"  // for throw_exception, RPY_THROW

#include <cereal/types/vector.hpp>

#include <algorithm>

using namespace rpy;
using namespace rpy::streams;


dimn_t CategoricalChannel::num_variants() const
{
    return m_variants.size();
}
string CategoricalChannel::label_suffix(dimn_t variant_no) const
{
    RPY_CHECK(variant_no < m_variants.size());
    return ":" + m_variants[variant_no];
}
dimn_t CategoricalChannel::variant_id_of_label(string_view label) const
{
    auto it = std::find(m_variants.begin(), m_variants.end(), label);
    if (it == m_variants.end()) {
        RPY_THROW(std::runtime_error,
                "unrecognised variant label for type categorical");
    }

    return static_cast<dimn_t>(it - m_variants.begin());
}

const std::vector<string>& CategoricalChannel::get_variants() const
{
    return m_variants;
}
StreamChannel& CategoricalChannel::add_variant(string variant_label)
{
    string label;
    if (variant_label.empty()) {
        label = std::to_string(m_variants.size());
    } else {
        label = variant_label;
    }

    auto var_begin = m_variants.begin();
    auto var_end = m_variants.end();
    auto found = std::find(var_begin, var_end, label);
    if (found != var_end) {
        RPY_THROW(std::runtime_error,
                  "variant with label " + label + " already exists");
    }
    m_variants.push_back(std::move(label));

    return *this;

}
StreamChannel& CategoricalChannel::insert_variant(string variant_label)
{
    string label;
    if (variant_label.empty()) {
        label = std::to_string(m_variants.size());
    } else {
        label = variant_label;
    }

    auto var_begin = m_variants.begin();
    auto var_end = m_variants.end();
    auto found = std::find(var_begin, var_end, label);
    if (found == var_end) {
        m_variants.push_back(std::move(label));
    }

    return *this;
}

#define RPY_EXPORT_MACRO ROUGHPY_STREAMS_EXPORT
#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::CategoricalChannel
#define RPY_SERIAL_DO_REGISTER
#include <roughpy/platform/serialization_instantiations.inl>

RPY_SERIAL_DYNAMIC_INIT(categorical_channel)
