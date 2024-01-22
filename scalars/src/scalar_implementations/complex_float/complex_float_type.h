//
// Created by user on 03/08/23.
//

#ifndef ROUGHPY_SCALARS_SRC_SCALAR_IMPLEMENTATIONS_COMPLEX_FLOAT_COMPLEX_FLOAT_TYPE_H_
#define ROUGHPY_SCALARS_SRC_SCALAR_IMPLEMENTATIONS_COMPLEX_FLOAT_COMPLEX_FLOAT_TYPE_H_

#include "standard_scalar_type.h"

namespace rpy {
namespace scalars {

class ComplexFloatType : public StandardScalarType<float_complex>
{
public:
    ComplexFloatType();

    std::unique_ptr<RandomGenerator>
    get_rng(const string& bit_generator, Slice<uint64_t> seed) const override;

    std::unique_ptr<BlasInterface> get_blas() const override;
};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_SCALAR_IMPLEMENTATIONS_COMPLEX_FLOAT_COMPLEX_FLOAT_TYPE_H_