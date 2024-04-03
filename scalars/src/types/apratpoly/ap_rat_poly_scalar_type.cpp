//
// Created by user on 06/11/23.
//

#include "ap_rat_poly_scalar_type.h"

#include "ap_rat_poly_type.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include "devices/host_device.h"
#include "random.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalar_implementations/poly_rational.h"
#include "types/aprational/ap_rational_scalar_type.h"


#include <algorithm>

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        ap_rat_poly_ring_characteristics{false, false, false, false};

APRatPolyType::APRatPolyType()
    : ScalarType(
              &devices::arbitrary_precision_rational_poly_type,
              devices::get_host_device(),
              ap_rat_poly_ring_characteristics
      )
{}

ScalarArray APRatPolyType::allocate(dimn_t count) const
{
    auto result = ScalarType::allocate(count);
    std::uninitialized_default_construct_n(
            static_cast<APPolyRat*>(result.mut_buffer().ptr()),
            count
    );
    return result;
}

void* APRatPolyType::allocate_single() const
{
    const auto access = this->lock();
    auto [pos, inserted] = m_allocations.insert(new APPolyRat());
    RPY_DBG_ASSERT(inserted);
    return *pos;
}

void APRatPolyType::free_single(void* ptr) const
{
    const auto access = this->lock();
    auto found = m_allocations.find(ptr);
    RPY_CHECK(found != m_allocations.end());
    delete static_cast<APPolyRat*>(*found);
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

const ScalarType* APRatPolyType::with_device(const devices::Device& device
) const
{
    return ScalarType::with_device(device);
}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<APPolyRat>::get() noexcept
// {
//     return APRatPolyType::get();
// }

const APRatPolyType scalars::arbitrary_precision_rational_polynomial_type;

const ScalarType* APRatPolyType::get() noexcept
{
    return &arbitrary_precision_rational_polynomial_type;
}
