//
// Created by sam on 24/07/24.
//

#ifndef DENSE_VECTOR_FACTORY_H
#define DENSE_VECTOR_FACTORY_H

#include "vector.h"

namespace rpy {
namespace algebra {

class DenseVectorFactory : public VectorFactory
{
    BasisPointer p_basis;
public:
    explicit DenseVectorFactory(BasisPointer basis)
        : p_basis(std::move(basis))
    {}

    Vector construct_empty() const override;
    Vector construct_from(const scalars::ScalarVector& base) const override;
    Vector construct_with_dim(dimn_t dimension) const override;
};

}// namespace algebra
}// namespace rpy

#endif// DENSE_VECTOR_FACTORY_H
