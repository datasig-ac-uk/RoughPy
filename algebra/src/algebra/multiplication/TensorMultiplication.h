//
// Created by sam on 19/09/24.
//

#ifndef TENSORMULTIPLICATION_H
#define TENSORMULTIPLICATION_H

#include "scalar_vector.h"

#include <roughpy/devices/buffer.h>

namespace rpy {
namespace algebra {

class TensorMultiplication
{
    devices::Buffer m_powers;
    devices::Buffer m_reverses;
    devices::Buffer m_offsets;

    deg_t m_width;
    deg_t m_depth;
    deg_t m_tile_letters;

public:
    TensorMultiplication(deg_t width, deg_t depth, deg_t tile_letters = 0);

    RPY_NO_DISCARD deg_t tile_letters() const noexcept
    {
        return m_tile_letters;
    }
    RPY_NO_DISCARD const devices::Buffer& powers() const noexcept
    {
        return m_powers;
    }
    RPY_NO_DISCARD const devices::Buffer& reverses() const noexcept
    {
        return m_reverses;
    }
    RPY_NO_DISCARD const devices::Buffer& offsets() const noexcept
    {
        return m_offsets;
    }

    void antipode(
            scalars::ScalarVector& result,
            const scalars::ScalarVector& arg
    ) const;
    void
    reflect(scalars::ScalarVector& result,
            const scalars::ScalarVector& arg) const;
};

}// namespace algebra
}// namespace rpy

#endif// TENSORMULTIPLICATION_H
