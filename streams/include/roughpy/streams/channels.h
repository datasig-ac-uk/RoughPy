#ifndef ROUGHPY_STREAMS_CHANNELS_H_
#define ROUGHPY_STREAMS_CHANNELS_H_

#include <roughpy/algebra/algebra_fwd.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/serialization.h>

#include <new>
#include <vector>

namespace rpy {
namespace streams {

enum struct ChannelType : uint8_t
{
    Increment = 0,
    Value = 1,
    Categorical = 2,
    Lie = 3,
};

enum struct StaticChannelType : uint8_t
{
    Value = 0,
    Categorical = 1
};

struct IncrementChannelInfo {
};
struct ValueChannelInfo {
    bool lead_lag = false;
};

struct CategoricalChannelInfo {
    std::vector<string> variants;
};
struct LieChannelInfo {
    deg_t width;
    deg_t depth;
    algebra::VectorType vtype;
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
    union
    {
        IncrementChannelInfo increment_info;
        ValueChannelInfo value_info;
        CategoricalChannelInfo categorical_info;
        LieChannelInfo lie_info;
    };

    template <typename T>
    static void inplace_construct(void* address, T&& value) noexcept(
            is_nothrow_constructible<remove_cv_ref_t<T>,
                                     decltype(value)>::value)
    {
        ::new (address) remove_cv_ref_t<T>(std::forward<T>(value));
    }

public:
    RPY_NO_DISCARD
    ChannelType type() const noexcept { return m_type; }

    RPY_NO_DISCARD dimn_t num_variants() const
    {
        switch (m_type) {
            case ChannelType::Increment: return 1;
            case ChannelType::Value: return value_info.lead_lag ? 2 : 1;
            case ChannelType::Categorical:
                return categorical_info.variants.size();
            case ChannelType::Lie: return lie_info.width;
        }
        RPY_UNREACHABLE();
    }

    RPY_NO_DISCARD
    string label_suffix(dimn_t variant_no) const;

    StreamChannel();
    StreamChannel(const StreamChannel& arg);
    StreamChannel(StreamChannel&& arg) noexcept;

    explicit StreamChannel(ChannelType type);

    explicit StreamChannel(IncrementChannelInfo&& info)
        : m_type(ChannelType::Increment), increment_info(std::move(info))
    {}

    explicit StreamChannel(ValueChannelInfo&& info)
        : m_type(ChannelType::Value), value_info(std::move(info))
    {}

    explicit StreamChannel(CategoricalChannelInfo&& info)
        : m_type(ChannelType::Categorical), categorical_info(std::move(info))
    {}

    explicit StreamChannel(LieChannelInfo&& info)
        : m_type(ChannelType::Lie), lie_info(std::move(info))
    {}

    ~StreamChannel();

    StreamChannel& operator=(const StreamChannel& other);
    StreamChannel& operator=(StreamChannel&& other) noexcept;

    StreamChannel& operator=(IncrementChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Increment;
        inplace_construct(&increment_info, std::move(info));
        return *this;
    }

    StreamChannel& operator=(ValueChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Value;
        inplace_construct(&value_info, std::move(info));
        return *this;
    }

    StreamChannel& operator=(CategoricalChannelInfo&& info)
    {
        this->~StreamChannel();
        m_type = ChannelType::Categorical;
        inplace_construct(&categorical_info, std::move(info));
        return *this;
    }

    RPY_NO_DISCARD
    dimn_t variant_id_of_label(string_view label) const;

    void set_lie_info(deg_t width, deg_t depth, algebra::VectorType vtype);
    void set_lead_lag(bool new_value)
    {
        RPY_CHECK(m_type == ChannelType::Value);
        value_info.lead_lag = new_value;
    }
    RPY_NO_DISCARD
    bool is_lead_lag() const
    {
        RPY_CHECK(m_type == ChannelType::Value);
        return value_info.lead_lag;
    }

    StreamChannel& add_variant(string variant_label);

    /**
     * @brief Insert a new variant if it doesn't already exist
     * @param variant_label variant label to insert.
     * @return referene to this channel.
     */
    StreamChannel& insert_variant(string variant_label);

    RPY_NO_DISCARD
    std::vector<string> get_variants() const;

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};

class RPY_EXPORT StaticChannel
{
    StaticChannelType m_type;
    union
    {
        ValueChannelInfo value_info;
        CategoricalChannelInfo categorical_info;
    };

    template <typename T>
    static void inplace_construct(void* address, T&& value) noexcept(
            is_nothrow_constructible<remove_cv_ref_t<T>,
                                     decltype(value)>::value)
    {
        ::new (address) remove_cv_ref_t<T>(std::forward<T>(value));
    }

public:
    StaticChannel();
    StaticChannel(const StaticChannel& other);
    StaticChannel(StaticChannel&& other) noexcept;

    explicit StaticChannel(ValueChannelInfo&& info)
        : m_type(StaticChannelType::Value), value_info(std::move(info))
    {}

    explicit StaticChannel(CategoricalChannelInfo&& info)
        : m_type(StaticChannelType::Categorical),
          categorical_info(std::move(info))
    {}

    ~StaticChannel();

    StaticChannel& operator=(const StaticChannel& other);
    StaticChannel& operator=(StaticChannel&& other) noexcept;

    RPY_NO_DISCARD
    StaticChannelType type() const noexcept { return m_type; }

    RPY_NO_DISCARD
    string label_suffix(dimn_t index) const;

    RPY_NO_DISCARD
    dimn_t num_variants() const noexcept;

    RPY_NO_DISCARD
    std::vector<string> get_variants() const;

    RPY_NO_DISCARD
    dimn_t variant_id_of_label(const string& label) const;

    StaticChannel& insert_variant(string new_variant);

    StaticChannel& add_variant(string new_variant);

    RPY_SERIAL_LOAD_FN();
    RPY_SERIAL_SAVE_FN();
};

RPY_SERIAL_SERIALIZE_FN_EXT(IncrementChannelInfo)
{
    (void) value;
    RPY_SERIAL_SERIALIZE_NVP("data", 0);
}

RPY_SERIAL_SERIALIZE_FN_EXT(ValueChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("lead_lag", value.lead_lag);
}

RPY_SERIAL_SERIALIZE_FN_EXT(CategoricalChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("variants", value.variants);
}

RPY_SERIAL_SERIALIZE_FN_EXT(LieChannelInfo)
{
    RPY_SERIAL_SERIALIZE_NVP("width", value.width);
    RPY_SERIAL_SERIALIZE_NVP("depth", value.depth);
    RPY_SERIAL_SERIALIZE_NVP("vector_type", value.vtype);
}

RPY_SERIAL_SAVE_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case ChannelType::Increment:
            RPY_SERIAL_SERIALIZE_NVP("increment", increment_info);
            break;
        case ChannelType::Value:
            RPY_SERIAL_SERIALIZE_NVP("value", value_info);
            break;
        case ChannelType::Categorical:
            RPY_SERIAL_SERIALIZE_NVP("categorical", categorical_info);
            break;
        case ChannelType::Lie: RPY_SERIAL_SERIALIZE_NVP("lie", lie_info); break;
    }
}

RPY_SERIAL_LOAD_FN_IMPL(StreamChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case ChannelType::Increment:
            new (&increment_info) IncrementChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("increment", increment_info);
            break;
        case ChannelType::Value:
            new (&value_info) ValueChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("value", value_info);
            break;
        case ChannelType::Categorical:
            new (&categorical_info) CategoricalChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("categorical", categorical_info);
            break;
        case ChannelType::Lie:
            new (&lie_info) LieChannelInfo;
            RPY_SERIAL_SERIALIZE_NVP("lie", lie_info);
            break;
    }
}

RPY_SERIAL_SAVE_FN_IMPL(StaticChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case StaticChannelType::Value:
            RPY_SERIAL_SERIALIZE_NVP("value", value_info);
            break;
        case StaticChannelType::Categorical:
            RPY_SERIAL_SERIALIZE_NVP("categorical", categorical_info);
            break;
    }
}

RPY_SERIAL_LOAD_FN_IMPL(StaticChannel)
{
    RPY_SERIAL_SERIALIZE_NVP("type", m_type);
    switch (m_type) {
        case StaticChannelType::Value:
            ::new (&value_info) ValueChannelInfo;
            break;
        case StaticChannelType::Categorical:
            ::new (&categorical_info) CategoricalChannelInfo;
            break;
    }
}

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_CHANNELS_H_
