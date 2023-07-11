#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_

#include "algebra_base.h"
#include "algebra_bundle.h"
#include "tensor_basis.h"

namespace rpy {
namespace algebra {

// extern template class AlgebraInterface<ShuffleTensor, TensorBasis>;

class RPY_EXPORT ShuffleTensorInterface
    : public AlgebraInterface<ShuffleTensor, TensorBasis>
{
    using base_t = AlgebraInterface<ShuffleTensor, TensorBasis>;

public:
    using base_t::base_t;
};

// extern template class AlgebraBase<ShuffleTensorInterface>;

class RPY_EXPORT ShuffleTensor : public AlgebraBase<ShuffleTensorInterface>
{
    using base_t = AlgebraBase<ShuffleTensorInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensor;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

class ShuffleTensorBundle;

// extern template class BundleInterface<ShuffleTensorBundle, ShuffleTensor,
//                                       ShuffleTensor>;

class RPY_EXPORT ShuffleTensorBundleInterface
    : public BundleInterface<ShuffleTensorBundle, ShuffleTensor, ShuffleTensor>
{
};

// extern template class AlgebraBundleBase<ShuffleTensorBundleInterface>;

class RPY_EXPORT ShuffleTensorBundle
    : public AlgebraBundleBase<ShuffleTensorBundleInterface>
{
    using base_t = AlgebraBundleBase<ShuffleTensorBundleInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensorBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

}// namespace algebra
}// namespace rpy
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensor,
        rpy::serial::specialization::member_serialize
)
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensorBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_FWD_H_
