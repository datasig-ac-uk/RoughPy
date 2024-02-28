#ifndef ROUGHPY_ALGEBRA_LIE_BUNDLE_H_
#define ROUGHPY_ALGEBRA_LIE_BUNDLE_H_

#include "algebra_bundle.h"
#include "interfaces/lie_bundle_interface.h"
#include "lie_basis.h"

#include <roughpy/platform/serialization.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

namespace rpy {
namespace algebra {

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE
        AlgebraBundleBase<LieBundleInterface>;


class ROUGHPY_ALGEBRA_EXPORT LieBundle : public AlgebraBundleBase<LieBundleInterface>
{
    using base_t = AlgebraBundleBase<LieBundleInterface>;

public:
    static constexpr AlgebraType s_alg_type = AlgebraType::LieBundle;

    using base_t::base_t;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_EXTERN_SERIALIZE_CLS(LieBundle)


RPY_SERIAL_SERIALIZE_FN_IMPL(LieBundle) { RPY_SERIAL_SERIALIZE_BASE(base_t); }

template <>
ROUGHPY_ALGEBRA_EXPORT typename LieBundle::basis_type
basis_setup_helper<LieBundle>::get(const context_pointer& ctx);

}// namespace algebra
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::LieBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_LIE_BUNDLE_H_
