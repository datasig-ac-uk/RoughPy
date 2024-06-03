//
// Created by sam on 3/30/24.
//

#include "ap_rational_type.h"

#include <roughpy/core/alloc.h>

#include <roughpy/devices/buffer.h>
#include <roughpy/devices/device_handle.h>

using namespace rpy;
using namespace rpy::devices;

using type = scalars::ArbitraryPrecisionRational;

rpy::devices::APRationalType::APRationalType()
    : Type("Rational",
           "Rational",
           {TypeCode::ArbitraryPrecisionRational,
            sizeof(scalars::ArbitraryPrecisionRational),
            alignof(scalars::ArbitraryPrecisionRational),
            1},
           traits_of<scalars::ArbitraryPrecisionRational>())
{
    const auto host = get_host_device();

    host->register_algorithm_drivers<HostDriversImpl, type, type>(this, this);

    auto& num_traits = setup_num_traits();

    num_traits.rational_type = this;
    num_traits.real_type = this;
    num_traits.imag_type = nullptr;

    num_traits.abs = math_fn_impls::abs_fn<type>;
    num_traits.real = math_fn_impls::real_fn<type>;
}

Buffer APRationalType::allocate(Device device, dimn_t count) const
{
    RPY_CHECK(device->is_host());
    return Type::allocate(device, count);
}
void* APRationalType::allocate_single() const
{
    auto* ptr = Type::allocate_single();
    construct_inplace(static_cast<scalars::ArbitraryPrecisionRational*>(ptr));
    return ptr;
}
void APRationalType::free_single(void* ptr) const
{
    std::destroy_at(static_cast<scalars::ArbitraryPrecisionRational*>(ptr));
    Type::free_single(ptr);
}
bool APRationalType::supports_device(const Device& device) const noexcept
{
    return device->is_host();
}
const APRationalType* APRationalType::get() noexcept
{
    static const APRationalType type;
    return &type;
}
void APRationalType::display(std::ostream& os, const void* ptr) const
{
    os << *static_cast<const type*>(ptr);
}
void APRationalType::copy(void* dst, const void* src, dimn_t count) const
{
    const auto* src_begin = static_cast<const type*>(src);
    const auto* src_end = src_begin + count;
    std::copy(src_begin, src_end, static_cast<type*>(dst));
}
void APRationalType::move(void* dst, void* src, dimn_t count) const
{
    const auto src_begin = std::make_move_iterator(static_cast<type*>(src));
    const auto src_end = src_begin + static_cast<idimn_t>(count);
    std::copy(src_begin, src_end, static_cast<type*>(dst));
}
ConstReference APRationalType::zero() const
{
    static const type zero{0};
    return ConstReference(&zero, this);
}
ConstReference APRationalType::one() const
{
    static const type one{1};
    return ConstReference(&one, this);
}
ConstReference APRationalType::mone() const
{
    static const type mone{-1};
    return ConstReference(&mone, this);
}

namespace {

struct InitializeAllTheFundamentals {
    InitializeAllTheFundamentals()
    {
        const auto* tp = APRationalType::get();
        devices::dtl::SupportRegistration<type, type>::register_support(tp);
        devices::dtl::register_type_support<type>(
                tp,
                devices::dtl::FundamentalTypesList()
        );

    }
};

}// namespace

static const InitializeAllTheFundamentals s_fundamentals{};