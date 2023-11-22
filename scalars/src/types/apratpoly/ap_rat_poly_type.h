//
// Created by user on 06/11/23.
//

#ifndef ROUGHPY_SCALARS_SRC_TYPES_APRATPOLY_AP_RAT_POLY_TYPE_H_
#define ROUGHPY_SCALARS_SRC_TYPES_APRATPOLY_AP_RAT_POLY_TYPE_H_

#include "scalar_type.h"

#include <unordered_set>

namespace rpy {
namespace scalars {

class APRatPolyType : ScalarType
{

    mutable std::unordered_set<void*> m_allocations;

public:

    APRatPolyType();

    RPY_NO_DISCARD ScalarArray allocate(dimn_t count) const override;
    RPY_NO_DISCARD void* allocate_single() const override;

    void free_single(void* ptr) const override;
    void convert_copy(ScalarArray& dst, const ScalarArray& src) const override;
    void assign(ScalarArray& dst, Scalar value) const override;
    const ScalarType* with_device(const devices::Device& device) const override;


    static const ScalarType* get() noexcept;
};




}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_TYPES_APRATPOLY_AP_RAT_POLY_TYPE_H_
