//
// Created by user on 08/10/23.
//

#include "interfaces/algebra_bundle_interface.h"
#include "interfaces/algebra_interface.h"
#include "interfaces/lie_bundle_interface.h"
#include "interfaces/lie_interface.h"
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/lie_bundle.h>

#include <roughpy/algebra/implementors/lie_bundle_impl.h>

namespace rpy {
namespace algebra {


template class RPY_EXPORT_INSTANTIATION AlgebraBundleBase<LieBundleInterface>;

template <>
typename LieBundle::basis_type
basis_setup_helper<LieBundle>::get(const context_pointer& ctx)
{
    return ctx->get_lie_basis();
}
}// namespace algebra
}// namespace rpy
