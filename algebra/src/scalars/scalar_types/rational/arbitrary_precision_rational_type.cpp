//
// Created by sam on 24/06/24.
//

#include "arbitrary_precision_rational_type.h"

#include <roughpy/core/smart_ptr.h>
#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>

#include <roughpy/device_support/fundamental_type.h>

using namespace rpy;
using namespace rpy::scalars;
using namespace scalars::implementations;

namespace math_fn_impls = devices::math_fn_impls;

using devices::dtl::AddInplace;
using devices::dtl::CompareEqual;
using devices::dtl::CompareGreater;
using devices::dtl::CompareGreaterEqual;
using devices::dtl::CompareLess;
using devices::dtl::CompareLessEqual;
using devices::dtl::Convert;
using devices::dtl::DivInplace;
using devices::dtl::MulInplace;
using devices::dtl::SubInplace;

namespace {}

namespace rpy { namespace devices {

template <> TypePtr get_type<ArbitraryPrecisionRational>()
{
    return ArbitraryPrecisionRationalType::get();
}

}}

ArbitraryPrecisionRationalType::ArbitraryPrecisionRationalType()
    : Type("rap",
           "Rational",
           {devices::TypeCode::ArbitraryPrecisionRational,
            sizeof(ArbitraryPrecisionRational),
            alignof(ArbitraryPrecisionRational),
            1},
           devices::traits_of<ArbitraryPrecisionRational>())
{
    using type = ArbitraryPrecisionRational;
    const auto& device = devices::get_host_device();
    device->register_algorithm_drivers<devices::HostDriversImpl, type, type>(
            this,
            this
    );

    auto& num_traits = setup_num_traits();

    num_traits.rational_type = this;
    num_traits.real_type = this;
    num_traits.imag_type = this;

    num_traits.abs = math_fn_impls::abs_fn<type>;
    num_traits.real = math_fn_impls::real_fn<type>;
    {
        auto support = this->update_support(*this);
        support->arithmetic.add_inplace = +[](void* out, const void* in) {
            (*static_cast<type*>(out)) += *static_cast<const type*>(in);
        };
        support->arithmetic.sub_inplace = +[](void* out, const void* in) {
            *static_cast<type*>(out) -= *static_cast<const type*>(in);
        };
        support->arithmetic.mul_inplace = +[](void* out, const void* in) {
            *static_cast<type*>(out) *= *static_cast<const type*>(in);
        };
        support->arithmetic.div_inplace = +[](void* out, const void* in) {
            *static_cast<type*>(out) /= *static_cast<const type*>(in);
        };

        support->comparison.equals = +[](const void* lhs, const void* rhs) {
            return *static_cast<const type*>(lhs)
                    == *static_cast<const type*>(rhs);
        };
        support->comparison.less = +[](const void* lhs, const void* rhs) {
            return *static_cast<const type*>(lhs)
                    < *static_cast<const type*>(rhs);
        };
        support->comparison.less_equal = +[](const void* lhs, const void* rhs) {
            return *static_cast<const type*>(lhs)
                    <= *static_cast<const type*>(rhs);
        };
        support->comparison.greater = +[](const void* lhs, const void* rhs) {
            return *static_cast<const type*>(lhs)
                    > *static_cast<const type*>(rhs);
        };
        support->comparison.greater_equal
                = +[](const void* lhs, const void* rhs) {
                      return *static_cast<const type*>(lhs)
                              >= *static_cast<const type*>(rhs);
                  };

        support->conversions.convert = +[](void* out, const void* in) {
            *static_cast<type*>(out) = *static_cast<const type*>(in);
        };
    }

    devices::dtl::register_type_support<type>(
            this,
            devices::dtl::FundamentalTypesList()
    );
}

devices::Buffer ArbitraryPrecisionRationalType::allocate(
        devices::Device device,
        dimn_t count
) const
{
    auto buffer = Type::allocate(device, count);
    std::uninitialized_default_construct_n(
            static_cast<ArbitraryPrecisionRational*>(buffer.ptr()),
            count
    );
    return buffer;
}
void* ArbitraryPrecisionRationalType::allocate_single() const
{
    return new ArbitraryPrecisionRational();
}
void ArbitraryPrecisionRationalType::free_single(void* ptr) const
{
    delete static_cast<ArbitraryPrecisionRational*>(ptr);
}
bool ArbitraryPrecisionRationalType::supports_device(
        const devices::Device& device
) const noexcept
{
    return Type::supports_device(device);
}
devices::TypeComparison
ArbitraryPrecisionRationalType::compare_with(const Type& other) const noexcept
{
    return Type::compare_with(other);
}
void ArbitraryPrecisionRationalType::copy(
        void* dst,
        const void* src,
        dimn_t count
) const
{
    const auto* begin = static_cast<const ArbitraryPrecisionRational*>(src);
    const auto* end = begin + count;
    auto* out = static_cast<ArbitraryPrecisionRational*>(dst);
    ranges::copy(begin, end, out);
}
void ArbitraryPrecisionRationalType::move(void* dst, void* src, dimn_t count)
        const
{
    auto* out = static_cast<ArbitraryPrecisionRational*>(dst);
    auto* in = static_cast<ArbitraryPrecisionRational*>(src);
    for (dimn_t i = 0; i < count; ++i) { out[i] = std::move(in[i]); }
}
void ArbitraryPrecisionRationalType::display(std::ostream& os, const void* ptr)
        const
{
    os << *static_cast<const ArbitraryPrecisionRational*>(ptr);
}
devices::ConstReference ArbitraryPrecisionRationalType::zero() const
{
    static const ArbitraryPrecisionRational zero{};
    return devices::ConstReference{this, &zero};
}
devices::ConstReference ArbitraryPrecisionRationalType::one() const
{
    static const ArbitraryPrecisionRational one{1};
    return devices::ConstReference{this, &one};
}
devices::ConstReference ArbitraryPrecisionRationalType::mone() const
{
    static const ArbitraryPrecisionRational mone{-1};
    return devices::ConstReference{this, &mone};
}

const ArbitraryPrecisionRationalType*
ArbitraryPrecisionRationalType::get() noexcept
{
    return new ArbitraryPrecisionRationalType();
}
