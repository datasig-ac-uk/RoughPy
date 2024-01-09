#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_H_

#include "algebra_bundle.h"
#include "free_tensor.h"
#include "tensor_basis.h"

namespace rpy {
namespace algebra {

template <typename, template <typename> class>
class FreeTensorBundleImplementation;

RPY_TEMPLATE_EXTERN template class RPY_EXPORT_TEMPLATE AlgebraBundleBase<
        FreeTensorBundleInterface,
        FreeTensorBundleImplementation>;

class ROUGHPY_ALGEBRA_EXPORT FreeTensorBundle : public AlgebraBundleBase<
                                            FreeTensorBundleInterface,
                                            FreeTensorBundleImplementation>
{
    using base_t = AlgebraBundleBase<
            FreeTensorBundleInterface,
            FreeTensorBundleImplementation>;

public:
    using base_t::base_t;

    static constexpr AlgebraType s_alg_type = AlgebraType::FreeTensorBundle;

    RPY_NO_DISCARD FreeTensorBundle exp() const;
    RPY_NO_DISCARD FreeTensorBundle log() const;
    //    RPY_NO_DISCARD
    //    FreeTensorBundle inverse() const;
    RPY_NO_DISCARD FreeTensorBundle antipode() const;
    FreeTensorBundle& fmexp(const FreeTensorBundle& other);

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_EXTERN_SERIALIZE_CLS(FreeTensorBundle)

RPY_SERIAL_SERIALIZE_FN_IMPL(FreeTensorBundle)
{
    RPY_SERIAL_SERIALIZE_BASE(base_t);
}

template <>
ROUGHPY_ALGEBRA_EXPORT typename FreeTensorBundle::basis_type
basis_setup_helper<FreeTensorBundle>::get(const context_pointer& ctx);


}// namespace algebra
}// namespace rpy
RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::algebra::FreeTensorBundle,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_H_
