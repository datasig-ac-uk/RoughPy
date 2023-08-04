//
// Created by user on 03/08/23.
//

#include "complex_double_type.h"

#include "standard_linalg.h"
using namespace rpy;
using namespace rpy::scalars;

ComplexDoubleType::ComplexDoubleType()
    : StandardScalarType<double_complex>("cf64", "DPComplex")
{}

std::unique_ptr<RandomGenerator> ComplexDoubleType::get_rng(
        const string& bit_generator, Slice<uint64_t> seed
) const
{
    // The default function throws an error.
    return ScalarType::get_rng(bit_generator, seed);
}
std::unique_ptr<BlasInterface> ComplexDoubleType::get_blas() const
{
    return std::make_unique<StandardLinearAlgebra<double_complex, double>>(this
    );
}
