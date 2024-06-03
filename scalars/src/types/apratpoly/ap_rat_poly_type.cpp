//
// Created by sam on 3/30/24.
//

#include "ap_rat_poly_type.h"


using namespace rpy;
using namespace rpy::devices;

using type = scalars::APPolyRat;

APRatPolyType::APRatPolyType()
    : Type("APRatPoly",
           "APRatPoly",
           {TypeCode::APRationalPolynomial, sizeof(type), alignof(type), 1},
           traits_of<type>())
{}

Buffer APRatPolyType::allocate(Device device, dimn_t count) const
{
    RPY_CHECK(device->is_host());
    auto buf = Type::allocate(device, count);

    return buf;
}
void* APRatPolyType::allocate_single() const
{
    auto* ptr = Type::allocate_single();
    construct_inplace(static_cast<APRatPolyType*>(ptr));
    return ptr;
}
void APRatPolyType::free_single(void* ptr) const
{
    std::destroy_at(static_cast<APRatPolyType*>(ptr));
    Type::free_single(ptr);
}
bool APRatPolyType::supports_device(const Device& device) const noexcept
{
    return device->is_host();
}

const APRatPolyType* APRatPolyType::get() noexcept
{
    static const APRatPolyType rational_type;
    return &rational_type;
}
void APRatPolyType::copy(void* dst, const void* src, dimn_t count) const
{
    const auto* src_begin = static_cast<const type*>(src);
    const auto* src_end = src_begin + count;
    std::copy(src_begin, src_end, static_cast<type*>(dst));
}
void APRatPolyType::move(void* dst, void* src, dimn_t count) const
{
    const auto src_begin = std::make_move_iterator(static_cast<type*>(src));
    const auto src_end = src_begin + static_cast<idimn_t>(count);
    std::copy(src_begin, src_end, static_cast<type*>(dst));
}
void APRatPolyType::display(std::ostream& os, const void* ptr) const
{
    os << *static_cast<const type*>(ptr);
}
ConstReference APRatPolyType::zero() const
{
    static const type zero {};
    return ConstReference(&zero, this);
}
ConstReference APRatPolyType::one() const
{
    static const type one {1};
    return ConstReference(&one, this);
}
ConstReference APRatPolyType::mone() const
{
    static const type mone{-1};
    return ConstReference(&mone, this);
}
namespace {

struct InitializeAllTheFundamentals {
    InitializeAllTheFundamentals()
    {
        const auto* tp = APRatPolyType::get();
        devices::dtl::SupportRegistration<type, type>::register_support(tp);
        devices::dtl::register_type_support<type>(
                tp,
                devices::dtl::FundamentalTypesList()
        );

    }
};

}// namespace

static const InitializeAllTheFundamentals s_fundamentals{};
