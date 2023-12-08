//
// Created by user on 06/11/23.
//

#include "ap_rational_type.h"
#include "scalar_array.h"
#include "scalar.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        ap_rational_ring_characteristics{true, true, false, false};

APRationalType::APRationalType()
    : ScalarType(
              "Rational",
              "Rational",
              alignof(rational_scalar_type),
              devices::get_host_device(),
              devices::type_info<rational_scalar_type>(),
              ap_rational_ring_characteristics
      )
{}
ScalarArray APRationalType::allocate(dimn_t count) const
{
    auto result = ScalarType::allocate(count);
    std::uninitialized_default_construct_n(static_cast<rational_scalar_type*>(result.mut_pointer()), count);
    return result;
}
void* APRationalType::allocate_single() const
{
    guard_type access(m_lock);
    auto [pos, inserted] = m_allocations.insert(new rational_scalar_type());
    RPY_DBG_ASSERT(inserted);
    return *pos;
}
void APRationalType::free_single(void* ptr) const
{
    guard_type access(m_lock);
    auto found = m_allocations.find(ptr);
    RPY_CHECK(found != m_allocations.end());
    delete static_cast<rational_scalar_type*>(ptr);
    m_allocations.erase(found);
}
void APRationalType::convert_copy(ScalarArray& dst, const ScalarArray& src)
        const
{
    ScalarType::convert_copy(dst, src);
}
void APRationalType::assign(ScalarArray& dst, Scalar value) const
{
    ScalarType::assign(dst, value);
}

const ScalarType* APRationalType::get() noexcept
{
    static const APRationalType type;
    return &type;
}

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalars::dtl::ScalarTypeOfImpl<rational_scalar_type>::get() noexcept
{
    return APRationalType::get();
}
