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
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_type.h>

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
class RPY_EXPORT StreamChannel
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
            scalars::ScalarPointer& dst, const scalars::ScalarPointer& src,
            dimn_t count
    ) const;

    template <typename T>
    void convert_input(scalars::ScalarPointer& dst, const T& single_data) const
    {
        convert_input(dst, {scalars::type_id_of<T>(), &single_data}, 1);
    }

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

RPY_SERIAL_SAVE_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    RPY_SERIAL_SERIALIZE_NVP(
            "dtype_id", (p_scalar_type == nullptr) ? "" : p_scalar_type->id()
    );
}
RPY_SERIAL_LOAD_FN_IMPL(StreamChannel) {
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    string id;
    RPY_SERIAL_SERIALIZE_NVP("dtype_id", id);
    if (!id.empty()) {
        p_scalar_type = scalars::get_type(id);
    }
}

}// namespace streams
}// namespace rpy



#endif// ROUGHPY_STREAMS_STREAM_CHANNEL_H
