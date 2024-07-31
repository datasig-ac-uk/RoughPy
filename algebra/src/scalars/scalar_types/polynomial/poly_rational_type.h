//
// Created by sam on 24/06/24.
//

#ifndef POLY_RATIONAL_TYPE_H
#define POLY_RATIONAL_TYPE_H

#include <roughpy/core/macros.h>
#include <roughpy/core/ranges.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/core.h>
#include <roughpy/devices/device_handle.h>
#include <roughpy/devices/type.h>
#include <roughpy/platform/alloc.h>

#include "poly_rational.h"

namespace rpy {
namespace scalars {
namespace implementations {

template <typename Sca>
class RPY_LOCAL PolyRationalType : public devices::Type
{
    using Poly = Polynomial<Sca>;
    PolyRationalType();

public:
    RPY_NO_DISCARD devices::Buffer
    allocate(devices::Device device, dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;
    void free_single(void* ptr) const override;
    RPY_NO_DISCARD bool supports_device(const devices::Device& device
    ) const noexcept override;
    RPY_NO_DISCARD bool convertible_to(const Type& dest_type
    ) const noexcept override;
    RPY_NO_DISCARD bool convertible_from(const Type& src_type
    ) const noexcept override;
    RPY_NO_DISCARD devices::TypeComparison compare_with(const Type& other
    ) const noexcept override;
    void copy(void* dst, const void* src, dimn_t count) const override;
    void move(void* dst, void* src, dimn_t count) const override;
    void display(std::ostream& os, const void* ptr) const override;
    RPY_NO_DISCARD devices::ConstReference zero() const override;
    RPY_NO_DISCARD devices::ConstReference one() const override;
    RPY_NO_DISCARD devices::ConstReference mone() const override;
};

template <typename Sca>
struct PolyTraits;

template <>
struct PolyTraits<Rational32> {
    static constexpr char id[] = "pr32";
    static constexpr char name[] = "PolyRational32";
    static constexpr devices::TypeInfo
            info{devices::TypeCode::Polynomial, 4, 4, 1};
    using type = Polynomial<Rational32>;
};

template <>
struct PolyTraits<Rational64> {
    static constexpr char id[] = "pr64";
    static constexpr char name[] = "PolyRational64";
    static constexpr devices::TypeInfo
            info{devices::TypeCode::Polynomial, 8, 8, 1};
    using type = Polynomial<Rational64>;
};

template <>
struct PolyTraits<ArbitraryPrecisionRational> {
    static constexpr char id[] = "aprp";
    static constexpr char name[] = "PolyRat";
    static constexpr devices::TypeInfo info{
            devices::TypeCode::APRationalPolynomial,
            sizeof(APPolyRat),
            alignof(APPolyRat),
            1
    };
    using type = APPolyRat;
};

template <typename Sca>
PolyRationalType<Sca>::PolyRationalType()
    : Type(PolyTraits<Sca>::id,
           PolyTraits<Sca>::name,
           PolyTraits<Sca>::info,
           devices::traits_of<typename PolyTraits<Sca>::type>())
{}
template <typename Sca>
devices::Buffer
PolyRationalType<Sca>::allocate(devices::Device device, dimn_t count) const
{
    return Type::allocate(device, count);
}
template <typename Sca>
void* PolyRationalType<Sca>::allocate_single() const
{
    return Type::allocate_single();
}
template <typename Sca>
void PolyRationalType<Sca>::free_single(void* ptr) const
{
    Type::free_single(ptr);
}
template <typename Sca>
bool PolyRationalType<Sca>::supports_device(const devices::Device& device
) const noexcept
{
    return Type::supports_device(device);
}
template <typename Sca>
bool PolyRationalType<Sca>::convertible_to(const Type& dest_type) const noexcept
{
    return Type::convertible_to(dest_type);
}
template <typename Sca>
bool PolyRationalType<Sca>::convertible_from(const Type& src_type
) const noexcept
{
    return Type::convertible_from(src_type);
}
template <typename Sca>
devices::TypeComparison PolyRationalType<Sca>::compare_with(const Type& other
) const noexcept
{
    return Type::compare_with(other);
}
template <typename Sca>
void PolyRationalType<Sca>::copy(void* dst, const void* src, dimn_t count) const
{
    Type::copy(dst, src, count);
}
template <typename Sca>
void PolyRationalType<Sca>::move(void* dst, void* src, dimn_t count) const
{
    Type::move(dst, src, count);
}
template <typename Sca>
void PolyRationalType<Sca>::display(std::ostream& os, const void* ptr) const
{
    Type::display(os, ptr);
}
template <typename Sca>
devices::ConstReference PolyRationalType<Sca>::zero() const
{
    return Type::zero();
}
template <typename Sca>
devices::ConstReference PolyRationalType<Sca>::one() const
{
    return Type::one();
}
template <typename Sca>
devices::ConstReference PolyRationalType<Sca>::mone() const
{
    return Type::mone();
}

extern template class PolyRationalType<Rational32>;
extern template class PolyRationalType<Rational64>;
extern template class PolyRationalType<ArbitraryPrecisionRational>;

using PolyRational32Type = PolyRationalType<Rational32>;
using PolyRational64Type = PolyRationalType<Rational64>;
using PolyRationalArbitraryPrecisionType
        = PolyRationalType<ArbitraryPrecisionRational>;

}// namespace implementations
}// namespace scalars
}// namespace rpy

#endif// POLY_RATIONAL_TYPE_H
