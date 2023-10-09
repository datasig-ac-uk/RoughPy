//
// Created by user on 09/10/23.
//

#include "implementors/algebra_bundle_impl.h"
#include "interfaces/shuffle_tensor_bundle_interface.h"
#include <roughpy/algebra/algebra_bundle.h>
#include <roughpy/algebra/algebra_bundle_base_impl.h>
#include <roughpy/algebra/shuffle_tensor_bundle.h>
#include <roughpy/algebra/tensor_basis.h>

#include <roughpy/algebra/implementors/shuffle_tensor_bundle_impl.h>

using namespace rpy;
using namespace rpy::algebra;

namespace rpy { namespace algebra {



template class RPY_EXPORT_INSTANTIATION
        AlgebraBundleBase<ShuffleTensorBundleInterface>;


}}


template <>
typename ShuffleTensorBundle::basis_type
basis_setup_helper<ShuffleTensorBundle>::get(const context_pointer& ctx)
{
    return ctx->get_tensor_basis();
}
