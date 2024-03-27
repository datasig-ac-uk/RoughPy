//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_APRATIONAL_AP_RATIONAL_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_APRATIONAL_AP_RATIONAL_TYPE_H_

#include "scalar_type.h"
#include "scalar_implementations/arbitrary_precision_rational.h"



namespace rpy {
namespace scalars {

class APRationalType : public ScalarType
{
    mutable std::unordered_set<void*> m_allocations;

public:

    APRationalType();

    ScalarArray allocate(dimn_t count) const override;
    void* allocate_single() const override;
    void free_single(void* ptr) const override;
    void convert_copy(ScalarArray& dst, const ScalarArray& src) const override;
    void assign(ScalarArray& dst, Scalar value) const override;


    static const ScalarType* get() noexcept;
};

RPY_LOCAL
extern const APRationalType arbitrary_precision_rational_type;


}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_APRATIONAL_AP_RATIONAL_TYPE_H_
