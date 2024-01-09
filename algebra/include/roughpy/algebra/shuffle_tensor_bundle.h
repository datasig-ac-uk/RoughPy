#ifndef ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_H_
#define ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_H_

#include "algebra_bundle.h"
#include "interfaces/shuffle_tensor_bundle_interface.h"

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {


RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
AlgebraBundleBase<ShuffleTensorBundleInterface>;


class ROUGHPY_ALGEBRA_EXPORT ShuffleTensorBundle
    : public AlgebraBundleBase<ShuffleTensorBundleInterface>
{
    using base_t = AlgebraBundleBase<ShuffleTensorBundleInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::ShuffleTensorBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_EXTERN_SERIALIZE_CLS(ShuffleTensorBundle)

RPY_SERIAL_SERIALIZE_FN_IMPL(ShuffleTensorBundle)
{
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

template <>
ROUGHPY_ALGEBRA_EXPORT typename ShuffleTensorBundle::basis_type
basis_setup_helper<ShuffleTensorBundle>::get(const context_pointer& ctx);

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::ShuffleTensorBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_SHUFFLE_TENSOR_BUNDLE_H_
