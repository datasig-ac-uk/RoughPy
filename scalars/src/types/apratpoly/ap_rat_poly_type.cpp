//
// Created by user on 06/11/23.
//

#include "ap_rat_poly_type.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>
#include <roughpy/platform/devices/host_device.h>

#include "scalar_array.h"
#include "scalar.h"
#include "scalar_types.h"
#include "random.h"

#include <algorithm>

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics ap_rat_poly_ring_characteristics{
        false,
        false,
        false,
        false
};

APRatPolyType::APRatPolyType()
    : ScalarType("RationalPoly",
                 "RationalPoly",
                 alignof(rational_poly_scalar),
                 devices::get_host_device(),
                 devices::type_info<rational_poly_scalar>(),
                 ap_rat_poly_ring_characteristics) {}


ScalarArray APRatPolyType::allocate(dimn_t count) const
{
    auto result = ScalarType::allocate(count);
    std::uninitialized_default_construct_n(static_cast<rational_poly_scalar*>(result.mut_pointer()), count);
    return result;
}

void* APRatPolyType::allocate_single() const
{
    guard_type access(m_lock);
    auto [pos, inserted] = m_allocations.insert(new rational_poly_scalar());
    RPY_DBG_ASSERT(inserted);
    return *pos;
}

void APRatPolyType::free_single(void* ptr) const
{
    guard_type access(m_lock);
    auto found = m_allocations.find(ptr);
    RPY_CHECK(found != m_allocations.end());
    delete static_cast<rational_poly_scalar*>(*found);
    m_allocations.erase(found);
}



void APRatPolyType::convert_copy(ScalarArray& dst, const ScalarArray& src) const
{
    ScalarType::convert_copy(dst, src);
}

void APRatPolyType::assign(ScalarArray& dst, Scalar value) const
{
    ScalarType::assign(dst, value);
}

const ScalarType* APRatPolyType::
with_device(const devices::Device& device) const
{
    return ScalarType::with_device(device);
}

const ScalarType* APRatPolyType::get() noexcept
{
    static const APRatPolyType type;
    return &type;
}

template <>
ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
scalars::dtl::ScalarTypeOfImpl<rational_poly_scalar>::get() noexcept
{
    return APRatPolyType::get();
}
