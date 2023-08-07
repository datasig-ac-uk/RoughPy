//
// Created by user on 03/08/23.
//

#include "complex_float_type.h"
#include "standard_linalg.h"

#include <roughpy/core/macros.h>

using namespace rpy;
using namespace rpy::scalars;

ComplexFloatType::ComplexFloatType()
    : StandardScalarType<float_complex>("cf32", "SPComplex")
{}

std::unique_ptr<RandomGenerator> ComplexFloatType::get_rng(
        const string& bit_generator, Slice<uint64_t> seed
) const
{
    RPY_THROW(
            std::runtime_error,
            "random number generation of complex "
            "numbers is not currently "
            "supported"
    );
}
std::unique_ptr<BlasInterface> ComplexFloatType::get_blas() const
{
    return std::make_unique<StandardLinearAlgebra<complex_float, float>>(this);
}
