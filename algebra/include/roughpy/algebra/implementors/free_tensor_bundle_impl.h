#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_IMPL_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_IMPL_H_

#include "algebra_bundle_impl.h"
#include "free_tensor_impl.h"
#include <roughpy/algebra/interfaces/free_tensor_bundle_interface.h>

namespace rpy {
namespace algebra {

template <typename FTBImpl, template <typename> class StorageModel>
class FreeTensorBundleImplementation : public AlgebraBundleImplementation<
                                               FreeTensorBundleInterface,
                                               FTBImpl,
                                               StorageModel>
{
    using base_t = AlgebraBundleImplementation<
            FreeTensorBundleInterface,
            FTBImpl,
            StorageModel>;

public:
    using base_t::base_t;

    FreeTensorBundle exp() const override;
    FreeTensorBundle log() const override;
    //    FreeTensorBundle inverse() const override;
    FreeTensorBundle antipode() const override;
    void fmexp(const FreeTensorBundle& other) override;
};

template <typename FTBImpl, template <typename> class StorageModel>
FreeTensorBundle
FreeTensorBundleImplementation<FTBImpl, StorageModel>::exp() const
{
    return FreeTensorBundle(
            FreeTensorBundleInterface::p_ctx,
            dtl::exp_wrapper(base_t::data())
    );
}
template <typename FTBImpl, template <typename> class StorageModel>
FreeTensorBundle
FreeTensorBundleImplementation<FTBImpl, StorageModel>::log() const
{
    return FreeTensorBundle(
            FreeTensorBundleInterface::p_ctx,
            dtl::log_wrapper(base_t::data())
    );
}
// template <typename FTBImpl, template <typename> class StorageModel>
// FreeTensorBundle
// FreeTensorBundleImplementation<FTBImpl, StorageModel>::inverse() const
//{
//     return FreeTensorBundle(
//             FreeTensorBundleInterface::p_ctx,
//             dtl::inverse_wrapper(base_t::data())
//     );
// }
template <typename FTBImpl, template <typename> class StorageModel>
FreeTensorBundle
FreeTensorBundleImplementation<FTBImpl, StorageModel>::antipode() const
{
    return FreeTensorBundle(
            FreeTensorBundleInterface::p_ctx,
            dtl::antipode_wrapper(base_t::data())
    );
}
template <typename FTBImpl, template <typename> class StorageModel>
void FreeTensorBundleImplementation<FTBImpl, StorageModel>::fmexp(
        const FreeTensorBundle& other
)
{
    base_t::data().fmexp_inplace(
            FreeTensorBundleImplementation::convert_argument(other)
    );
}
}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_BUNDLE_IMPL_H_
