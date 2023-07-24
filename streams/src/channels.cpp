// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 04/07/23.
//
#include <roughpy/streams/channels.h>

#include <algorithm>

using namespace rpy;
using namespace streams;

StreamChannel::StreamChannel()
    : m_type(ChannelType::Increment), increment_info()
{}
StreamChannel::StreamChannel(const StreamChannel& arg) : m_type(arg.m_type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, arg.increment_info);
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, arg.value_info);
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, arg.categorical_info);
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, arg.lie_info);
            break;
    }
}
StreamChannel::StreamChannel(StreamChannel&& arg) noexcept : m_type(arg.m_type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, arg.increment_info);
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, arg.value_info);
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, arg.categorical_info);
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, arg.lie_info);
            break;
    }
}

StreamChannel::StreamChannel(ChannelType type) : m_type(type)
{
    switch (m_type) {
        case ChannelType::Increment:
            inplace_construct(&increment_info, IncrementChannelInfo());
            break;
        case ChannelType::Value:
            inplace_construct(&value_info, ValueChannelInfo());
            break;
        case ChannelType::Categorical:
            inplace_construct(&categorical_info, CategoricalChannelInfo());
            break;
        case ChannelType::Lie:
            inplace_construct(&lie_info, LieChannelInfo());
            break;
    }
}

StreamChannel& StreamChannel::operator=(const StreamChannel& other)
{
    if (&other != this) {
        this->~StreamChannel();
        m_type = other.m_type;
        switch (m_type) {
            case ChannelType::Increment:
                inplace_construct(&increment_info, other.increment_info);
                break;
            case ChannelType::Value:
                inplace_construct(&value_info, other.value_info);
                break;
            case ChannelType::Categorical:
                inplace_construct(&categorical_info, other.categorical_info);
                break;
            case ChannelType::Lie:
                inplace_construct(&lie_info, other.lie_info);
                break;
        }
    }
    return *this;
}
StreamChannel& StreamChannel::operator=(StreamChannel&& other) noexcept
{
    if (&other != this) {
        this->~StreamChannel();
        m_type = other.m_type;
        switch (m_type) {
            case ChannelType::Increment:
                inplace_construct(&increment_info,
                                  std::move(other.increment_info));
                break;
            case ChannelType::Value:
                inplace_construct(&value_info, std::move(other.value_info));
                break;
            case ChannelType::Categorical:
                inplace_construct(&categorical_info,
                                  std::move(other.categorical_info));
                break;
            case ChannelType::Lie:
                inplace_construct(&lie_info, std::move(other.lie_info));
                break;
        }
    }
    return *this;
}



StreamChannel::~StreamChannel()
{
    switch (m_type) {
        case ChannelType::Increment:
            increment_info.~IncrementChannelInfo();
            break;
        case ChannelType::Value: value_info.~ValueChannelInfo(); break;
        case ChannelType::Categorical:
            categorical_info.~CategoricalChannelInfo();
            break;
        case ChannelType::Lie: lie_info.~LieChannelInfo(); break;
    }
}



string StreamChannel::label_suffix(dimn_t variant_no) const
{
    switch (m_type) {
        case ChannelType::Increment: return "";
        case ChannelType::Value:
            if (value_info.lead_lag) {
                RPY_CHECK(variant_no < 2);
                return (variant_no == 0) ? ":lead" : ":lag";
            } else {
                return "";
            };
        case ChannelType::Categorical:
            RPY_CHECK(variant_no < categorical_info.variants.size());
            return ":" + categorical_info.variants[variant_no];
        case ChannelType::Lie:
            RPY_CHECK(variant_no < static_cast<dimn_t>(lie_info.width));
            return ":" + std::to_string(variant_no + 1);
    }
    RPY_UNREACHABLE_RETURN({});
}

void StreamChannel::set_lie_info(deg_t width, deg_t depth,
                                 algebra::VectorType vtype)
{
    RPY_CHECK(m_type == ChannelType::Lie);
    lie_info.width = width;
    lie_info.depth = depth;
    lie_info.vtype = vtype;
}

dimn_t StreamChannel::variant_id_of_label(string_view label) const
{
    switch (m_type) {
        case ChannelType::Increment: return 0;
        case ChannelType::Value:
            if (value_info.lead_lag) {
                if (label == "lead") {
                    return 0;
                } else if (label == "lag") {
                    return 1;
                } else {
                    RPY_THROW(std::runtime_error,
                            "unrecognised variant label for type value");
                }
            } else {
                return 0;
            }
        case ChannelType::Categorical: break;
        case ChannelType::Lie:
            deg_t i = std::stoi(string(label));
            RPY_CHECK(i < lie_info.width);
            return i;
    }

    auto it = std::find(categorical_info.variants.begin(),
                        categorical_info.variants.end(), label);
    if (it == categorical_info.variants.end()) {
        RPY_THROW(std::runtime_error,
                "unrecognised variant label for type categorical");
    }

    return static_cast<dimn_t>(it - categorical_info.variants.begin());
}

StreamChannel& StreamChannel::add_variant(string variant_label)
{
    RPY_CHECK(m_type == ChannelType::Categorical);

    if (variant_label.empty()) {
        variant_label = std::to_string(categorical_info.variants.size());
    }

    auto found = std::find(categorical_info.variants.begin(),
                           categorical_info.variants.end(), variant_label);
    if (found != categorical_info.variants.end()) {
        RPY_THROW(std::runtime_error,"variant with label " + variant_label
                                 + " already exists");
    }

    categorical_info.variants.push_back(std::move(variant_label));
    return *this;
}
StreamChannel& StreamChannel::insert_variant(string variant_label)
{
    RPY_CHECK(m_type == ChannelType::Categorical);

    if (variant_label.empty()) {
        variant_label = std::to_string(categorical_info.variants.size());
    }

    auto var_begin = categorical_info.variants.begin();
    auto var_end = categorical_info.variants.end();

    auto found = std::find(var_begin, var_end, variant_label);
    if (found == var_end) {
        categorical_info.variants.push_back(std::move(variant_label));
    }

    return *this;
}

std::vector<string> StreamChannel::get_variants() const
{
    std::vector<string> variants;
    switch (m_type) {
        case ChannelType::Increment: break;
        case ChannelType::Value:
            if (value_info.lead_lag) {
                variants.push_back("lead");
                variants.push_back("lag");
            }
            break;
        case ChannelType::Categorical:
            variants = categorical_info.variants;
            break;
        case ChannelType::Lie:
            variants.reserve(lie_info.width);
            for (deg_t i = 0; i < lie_info.width; ++i) {
                variants.push_back(std::to_string(i));
            }
            break;
    }
    return variants;
}

///////////////////////////////////////////////////////////////////////////////
// Static channel
///////////////////////////////////////////////////////////////////////////////

StaticChannel::StaticChannel() : m_type(StaticChannelType::Value)
{
    inplace_construct(&value_info, ValueChannelInfo());
}
StaticChannel::StaticChannel(const StaticChannel& other) : m_type(other.m_type)
{
    switch (m_type) {
        case StaticChannelType::Value:
            inplace_construct(&value_info, other.value_info);
            break;
        case StaticChannelType::Categorical:
            inplace_construct(&categorical_info, other.categorical_info);
            break;
    }
}

StaticChannel::StaticChannel(StaticChannel&& other) noexcept
        : m_type(other.m_type)
{
    switch (m_type) {
        case StaticChannelType::Value:
            inplace_construct(&value_info, std::move(other.value_info));
            break;
        case StaticChannelType::Categorical:
            inplace_construct(&categorical_info,
                              std::move(other.categorical_info));
            break;
    }
}
StaticChannel::~StaticChannel()
{
    switch (m_type) {
        case StaticChannelType::Value: value_info.~ValueChannelInfo(); break;
        case StaticChannelType::Categorical:
            categorical_info.~CategoricalChannelInfo();
            break;
    }
}

StaticChannel& StaticChannel::operator=(const StaticChannel& other)
{
    if (&other != this) {
        this->~StaticChannel();
        m_type = other.m_type;
        switch (m_type) {
            case StaticChannelType::Value:
                inplace_construct(&value_info, other.value_info);
                break;
            case StaticChannelType::Categorical:
                inplace_construct(&categorical_info, other.categorical_info);
                break;
        }
    }
    return *this;
}
StaticChannel& StaticChannel::operator=(StaticChannel&& other) noexcept
{
    if (&other != this) {
        this->~StaticChannel();
        m_type = other.m_type;
        switch (m_type) {
            case StaticChannelType::Value:
                inplace_construct(&value_info, std::move(other.value_info));
                break;
            case StaticChannelType::Categorical:
                inplace_construct(&categorical_info,
                                  std::move(other.categorical_info));
                break;
        }
    }
    return *this;
}
string StaticChannel::label_suffix(dimn_t index) const
{
    switch (m_type) {
        case StaticChannelType::Value: return {};
        case StaticChannelType::Categorical: {
            return categorical_info.variants[index];
        }
    }
    RPY_UNREACHABLE_RETURN({});
}
dimn_t StaticChannel::num_variants() const noexcept
{
    switch (m_type) {
        case StaticChannelType::Value: return 1;
        case StaticChannelType::Categorical:
            return categorical_info.variants.size();
    }
    RPY_UNREACHABLE_RETURN(0);
}
std::vector<string> StaticChannel::get_variants() const
{
    switch (m_type) {
        case StaticChannelType::Value: return {};
        case StaticChannelType::Categorical: return categorical_info.variants;
    }
    RPY_UNREACHABLE_RETURN({});
}
dimn_t StaticChannel::variant_id_of_label(const string& label) const
{
    switch (m_type) {
        case StaticChannelType::Value: return 0;
        case StaticChannelType::Categorical: {
            const auto begin = categorical_info.variants.begin();
            const auto end = categorical_info.variants.end();
            const auto found = std::find(begin, end, label);
            if (found == end) {
                RPY_THROW(std::runtime_error,"label " + label
                                         + " not a valid "
                                           "variant of this "
                                           "channel");
            }
            return static_cast<dimn_t>(found - begin);
        }
    }
    RPY_UNREACHABLE_RETURN(0);
}
StaticChannel& StaticChannel::insert_variant(string new_variant)
{
    RPY_CHECK(m_type == StaticChannelType::Categorical);
    const auto begin = categorical_info.variants.begin();
    const auto end = categorical_info.variants.end();
    const auto found = std::find(begin, end, new_variant);
    if (found == end) {
        categorical_info.variants.push_back(std::move(new_variant));
    }
    return *this;
}
StaticChannel& StaticChannel::add_variant(string new_variant)
{
    RPY_CHECK(m_type == StaticChannelType::Categorical);
    const auto begin = categorical_info.variants.begin();
    const auto end = categorical_info.variants.end();
    const auto found = std::find(begin, end, new_variant);

    RPY_CHECK(found == end);
    categorical_info.variants.push_back(std::move(new_variant));
    return *this;
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StreamChannel
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME IncrementChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME ValueChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME CategoricalChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_EXTERNAL rpy::streams
#define RPY_SERIAL_IMPL_CLASSNAME LieChannelInfo

#include <roughpy/platform/serialization_instantiations.inl>

#define RPY_SERIAL_IMPL_CLASSNAME rpy::streams::StaticChannel
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
