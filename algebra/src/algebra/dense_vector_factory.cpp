//
// Created by sam on 24/07/24.
//

#include "dense_vector_factory.h"

#include "vector.h"


using namespace rpy;
using namespace rpy::algebra;

Vector DenseVectorFactory::construct_empty() const
{
    return Vector(p_basis, p_type);
}
Vector DenseVectorFactory::construct_from(const scalars::ScalarVector& base
) const
{
    return Vector(p_basis, p_type);
}
Vector DenseVectorFactory::construct_with_dim(dimn_t dimension) const
{
    return Vector(p_basis, p_type);
}
