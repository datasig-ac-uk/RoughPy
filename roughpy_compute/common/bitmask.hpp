#ifndef ROUGHPY_COMPUTE_COMMON_BITMASK_HPP
#define ROUGHPY_COMPUTE_COMMON_BITMASK_HPP

#include <cassert>
#include <type_traits>

namespace rpy::compute {

template <typename Int>
class BitMask
{
    using UInt = std::make_unsigned_t<Int>;
    UInt base_ = 0;

    constexpr BitMask() = default;

    constexpr explicit BitMask(int n_bits)
        : base_((UInt{1} << n_bits) - UInt{1})
    {}

    constexpr UInt operator[](int idx) const noexcept
    {
        assert(idx < 8 * sizeof(UInt));
        return (base_ >> idx) & UInt{1};
    }

    constexpr BitMask& operator++() noexcept
    {
        ++base_;
        return *this;
    }

    constexpr BitMask operator++(int) noexcept
    {
        BitMask tmp = *this;
        ++(*this);
        return tmp;
    }

    friend constexpr bool
    operator<(const BitMask& lhs, const BitMask& rhs) noexcept
    {
        return lhs.base_ < rhs.base_;
    }

    friend constexpr bool
    operator==(const BitMask& lhs, const BitMask& rhs) noexcept
    {
        return lhs.base_ == rhs.base_;
    }
};

}

#endif// ROUGHPY_COMPUTE_COMMON_BITMASK_HPP
