#ifndef STREAM_METADATA_H
#define STREAM_METADATA_H

#include <vector>

#include "roughpy/core/types.h"

#include "roughpy/algebra/context_fwd.h"
#include "roughpy/algebra/context.h"

#include "roughpy/intervals/dyadic_interval.h"
#include "roughpy/intervals/real_interval.h"


namespace rpy {
namespace streams {

class StreamMetadataBuilder;


RPY_NO_DISCARD
inline resolution_t param_to_resolution(param_t arg) noexcept
{
    int exponent;
    frexp(arg, &exponent);
    /*
     * frexp returns fractional part in the range [0.5, 1), so the correct power
     * of 2 is actually exponent - 1.\
     */
    return -std::min(0, exponent - 1);
}


class StreamMetadata
{
    intervals::RealInterval m_domain;
    std::vector<string> m_channel_names;

    algebra::context_pointer p_default_context;

    deg_t m_resolution = 8;
    intervals::IntervalType m_interval_type = intervals::IntervalType::Clopen;


    friend StreamMetadataBuilder;

public:
    StreamMetadata() = default;


    StreamMetadata(intervals::RealInterval domain,
                   std::vector<string> channels,
                   algebra::context_pointer context,
                   deg_t resolution,
                   intervals::IntervalType interval_type =
                           intervals::IntervalType::Clopen
    )
        : m_domain(std::move(domain)), m_channel_names(std::move(channels)),
          p_default_context(std::move(context)), m_resolution(resolution),
          m_interval_type(interval_type) {}

    const intervals::RealInterval& domain() const noexcept { return m_domain; }

    span<const string> channel_names() const noexcept
    {
        return {m_channel_names.data(), m_channel_names.size()};
    }

    const algebra::context_pointer& default_context() const noexcept
    {
        return p_default_context;
    }

    const scalars::ScalarType* scalar_type() const noexcept
    {
        return p_default_context->ctype();
    }

    deg_t resolution() const noexcept { return m_resolution; }

    intervals::IntervalType interval_type() const noexcept
    {
        return m_interval_type;
    }

    static StreamMetadataBuilder builder(
        const StreamMetadata* base = nullptr) noexcept;

};


class StreamMetadataBuilder
{
    std::shared_ptr<StreamMetadata> p_metadata;
    const scalars::ScalarType* p_ctype = nullptr;
    deg_t m_default_depth = 2;

public:
    explicit StreamMetadataBuilder(const StreamMetadata* existing = nullptr);


    StreamMetadataBuilder& add_channel(string channel_name);

    StreamMetadataBuilder& set_domain(intervals::RealInterval domain) noexcept
    {
        p_metadata->m_domain = std::move(domain);
        return *this;
    }

    StreamMetadataBuilder& set_resolution_from_increment_min(
        param_t min_increment)
    {
        RPY_CHECK_GT(min_increment, 0.0);
        resolution_t resolution = param_to_resolution(min_increment);
        return set_resolution(resolution);
    }

    StreamMetadataBuilder& set_resolution(resolution_t resolution) noexcept
    {
        p_metadata->m_resolution = resolution;
        return *this;
    }

    StreamMetadataBuilder& set_depth(deg_t depth) noexcept
    {
        m_default_depth = depth;
        return *this;
    }

    StreamMetadataBuilder& set_scalar_type(
        const scalars::ScalarType* ctype) noexcept
    {
        p_ctype = ctype;
        return *this;
    }

    StreamMetadataBuilder& set_context(algebra::context_pointer ctx) noexcept
    {
        p_metadata->p_default_context = std::move(ctx);
        return *this;
    }

    std::shared_ptr<StreamMetadata> build();
};


inline StreamMetadataBuilder StreamMetadata::builder(
    const StreamMetadata* base) noexcept { return StreamMetadataBuilder(base); }

}// streams
}// rpy

#endif //STREAM_METADATA_H