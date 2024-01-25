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

#ifndef ROUGHPY_STREAMS_STREAM_CHANNEL_H
#define ROUGHPY_STREAMS_STREAM_CHANNEL_H

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/core/alloc.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/scalars/scalar_type.h>

#include "roughpy_streams_export.h"

namespace rpy {
namespace streams {

enum struct ChannelType : uint8_t
{
    Increment = 0,
    Value = 1,
    Categorical = 2,
    Lie = 3,
};

/**
 * @brief Abstract description of a single channel of data.
 *
 * A stream channel contains metadata about one particular channel of
 * incoming raw data. The data can take one of four possible forms:
 * increment data, value data, categorical data, or lie data. Each form
 * of data has its own set of required and optional metadata, including
 * the number and labels of a categorical variable. These objects are
 * typically accessed via a schema, which maintains the collection of
 * all channels associated with a stream.
 */
class ROUGHPY_STREAMS_EXPORT StreamChannel
{
    ChannelType m_type;
    const scalars::ScalarType* p_scalar_type = nullptr;

protected:
    explicit StreamChannel(
            ChannelType type, const scalars::ScalarType* channel_dtype
    )
        : m_type(type), p_scalar_type(channel_dtype)
    {}

public:

    StreamChannel() : m_type(ChannelType::Increment), p_scalar_type(nullptr)
    {}

    RPY_NO_DISCARD ChannelType type() const noexcept { return m_type; }

    RPY_NO_DISCARD virtual dimn_t num_variants() const;

    RPY_NO_DISCARD virtual string label_suffix(dimn_t variant_no) const;

    virtual ~StreamChannel();

    RPY_NO_DISCARD virtual dimn_t variant_id_of_label(string_view label) const;

    virtual void
    set_lie_info(deg_t width, deg_t depth, algebra::VectorType vtype);

    virtual void set_lead_lag(bool new_value);

    RPY_NO_DISCARD virtual bool is_lead_lag() const;

    virtual void convert_input(
            scalars::ScalarArray& dst, const scalars::ScalarArray& src
    ) const;
//
//    template <typename T>
//    void convert_input(scalars::ScalarPointer& dst, const T& single_data) const
//    {
//        convert_input(dst, {scalars::type_id_of<T>(), &single_data}, 1);
//    }

    virtual StreamChannel& add_variant(string variant_label);

    /**
     * @brief Insert a new variant if it doesn't already exist
     * @param variant_label variant label to insert.
     * @return referene to this channel.
     */
    virtual StreamChannel& insert_variant(string variant_label);

    RPY_NO_DISCARD virtual const std::vector<string>& get_variants() const;

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};

#ifdef RPY_COMPILING_STREAMS
RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(StreamChannel)
RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(StreamChannel)
#else
RPY_SERIAL_EXTERN_LOAD_CLS_IMP(StreamChannel)
RPY_SERIAL_EXTERN_SAVE_CLS_IMP(StreamChannel)
#endif

RPY_SERIAL_SAVE_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    RPY_SERIAL_SERIALIZE_NVP(
            "dtype_id", (p_scalar_type == nullptr) ? "" : string(p_scalar_type->id())
    );
}
RPY_SERIAL_LOAD_FN_IMPL(StreamChannel) {
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    string id;
    RPY_SERIAL_SERIALIZE_NVP("dtype_id", id);
    if (!id.empty()) {
        auto tp_o = scalars::get_type(id);
        RPY_CHECK(tp_o);
        p_scalar_type = *tp_o;
    }
}


RPY_SERIAL_SAVE_FN_EXT(ChannelType) {
    RPY_SERIAL_SERIALIZE_BARE(static_cast<uint8_t>(value));
}

RPY_SERIAL_LOAD_FN_EXT(ChannelType) {
    uint8_t tmp;
    RPY_SERIAL_SERIALIZE_BARE(tmp);
    value = static_cast<ChannelType>(tmp);
}

}// namespace streams
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::streams::ChannelType,
                            rpy::serial::specialization::non_member_load_save);

#endif// ROUGHPY_STREAMS_STREAM_CHANNEL_H
