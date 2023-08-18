#ifndef ROUGHPY_STREAMS_PARAMETRIZATION_H_
#define ROUGHPY_STREAMS_PARAMETRIZATION_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/intervals/interval.h>
#include <roughpy/intervals/real_interval.h>
#include <roughpy/platform/serialization.h>

namespace rpy {
namespace streams {

class RPY_EXPORT Parameterization
{
    param_t m_param_offset = 0.0;
    param_t m_param_scaling = 1.0;

    bool b_add_to_channels = false;
    idimn_t m_is_channel = -1;

public:
    virtual ~Parameterization();

    void add_as_channel() noexcept { b_add_to_channels = true; }

    RPY_NO_DISCARD bool is_channel() const noexcept
    {
        return b_add_to_channels || (m_is_channel >= 0);
    }

    RPY_NO_DISCARD bool needs_adding() const noexcept
    {
        return b_add_to_channels && m_is_channel < 0;
    }

    RPY_NO_DISCARD intervals::RealInterval
    reparametrize(const intervals::RealInterval& arg) const
    {
        return {m_param_offset + m_param_scaling * arg.inf(),
                m_param_offset + m_param_scaling * arg.sup(), arg.type()};
    }

    RPY_NO_DISCARD virtual intervals::RealInterval
    convert_parameter_interval(const intervals::Interval& arg) const;
};

}// namespace streams
}// namespace rpy

#endif// ROUGHPY_STREAMS_PARAMETRIZATION_H_
