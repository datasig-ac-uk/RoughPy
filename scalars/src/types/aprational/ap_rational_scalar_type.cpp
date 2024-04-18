//
// Created by user on 06/11/23.
//

#include "ap_rational_scalar_type.h"
#include "scalar.h"
#include "scalar_array.h"
#include "scalar_implementations/arbitrary_precision_rational.h"

#include "ap_rational_type.h"

using namespace rpy;
using namespace rpy::scalars;

static constexpr RingCharacteristics
        ap_rational_ring_characteristics{true, true, false, false};

APRationalScalarType::APRationalScalarType()
    : ScalarType(
              devices::APRationalType::get(),
              devices::get_host_device(),
              ap_rational_ring_characteristics
      )
{}
ScalarArray APRationalScalarType::allocate(dimn_t count) const
{
    auto result = ScalarType::allocate(count);
    std::uninitialized_default_construct_n(
            static_cast<ArbitraryPrecisionRational*>(result.mut_buffer().ptr()),
            count
    );
    return result;
}
void* APRationalScalarType::allocate_single() const
{
    const auto access = this->lock();
    auto [pos, inserted]
            = m_allocations.insert(new ArbitraryPrecisionRational());
    RPY_DBG_ASSERT(inserted);
    return *pos;
}
void APRationalScalarType::free_single(void* ptr) const
{
    const auto access = this->lock();
    auto found = m_allocations.find(ptr);
    RPY_CHECK(found != m_allocations.end());
    delete static_cast<ArbitraryPrecisionRational*>(ptr);
    m_allocations.erase(found);
}
void APRationalScalarType::convert_copy(
        ScalarArray& dst,
        const ScalarArray& src
) const
{
    ScalarType::convert_copy(dst, src);
}
void APRationalScalarType::assign(ScalarArray& dst, Scalar value) const
{
    ScalarType::assign(dst, value);
}

// template <>
// ROUGHPY_SCALARS_EXPORT optional<const ScalarType*>
// scalars::dtl::ScalarTypeOfImpl<ArbitraryPrecisionRational>::get() noexcept
// {
//     return APRationalType::get();
// }


const ScalarType* APRationalScalarType::get() noexcept
{
    static const APRationalScalarType rational_type;
    return &rational_type;
}
