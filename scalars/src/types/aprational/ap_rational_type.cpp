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
    return ScalarType::allocate(count);
}
void* APRationalType::allocate_single() const
{
    return ScalarType::allocate_single();
}
void APRationalType::free_single(void* ptr) const
{
    ScalarType::free_single(ptr);
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
