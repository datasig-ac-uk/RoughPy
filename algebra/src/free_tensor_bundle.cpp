//
// Created by user on 08/10/23.
//

#include <roughpy/algebra/implementors/algebra_bundle_impl.h>
#include <roughpy/algebra/interfaces/free_tensor_bundle_interface.h>
#include <roughpy/algebra/free_tensor.h>

using namespace rpy::algebra;

namespace rpy {
namespace algebra {



template class RPY_EXPORT_INSTANTIATION AlgebraBundleBase<
        FreeTensorBundleInterface,
        FreeTensorBundleImplementation>;

}// namespace algebra
}// namespace rpy

FreeTensorBundle FreeTensorBundle::exp() const
{
    if (p_impl) { return p_impl->exp(); }
    return FreeTensorBundle();
}
FreeTensorBundle FreeTensorBundle::log() const
{
    if (p_impl) { return p_impl->log(); }
    return FreeTensorBundle();
}
// FreeTensorBundle FreeTensorBundle::inverse() const
//{
//     if (p_impl) { return p_impl->inverse(); }
//     return FreeTensorBundle();
// }
FreeTensorBundle FreeTensorBundle::antipode() const
{
    if (p_impl) { return p_impl->antipode(); }
    return FreeTensorBundle();
}
FreeTensorBundle& FreeTensorBundle::fmexp(const FreeTensorBundle& other)
{
    if (p_impl && other.p_impl) { p_impl->fmexp(other); }
    return *this;
}

template <>
typename FreeTensorBundle::basis_type
basis_setup_helper<FreeTensorBundle>::get(const context_pointer& ctx)
{
    return ctx->get_tensor_basis();
}
