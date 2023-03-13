#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_

#include "algebra_impl.h"
#include "free_tensor.h"

namespace rpy {
namespace algebra {

template <typename FTImpl, template <typename> class StorageModel>
class FreeTensorImplementation : public AlgebraImplementation<FreeTensorInterface, FTImpl, StorageModel> {
    using base_t = AlgebraImplementation<FreeTensorInterface, FTImpl, StorageModel>;

public:
    using base_t::base_t;

    using base_t::add_inplace;
    using base_t::assign;
    using base_t::clone;
    using base_t::equals;
    using base_t::get;
    using base_t::mul_inplace;
    using base_t::sub_inplace;
    using base_t::uminus;
    using base_t::zero_like;

    FreeTensor exp() const override;
    FreeTensor log() const override;
    FreeTensor inverse() const override;
    FreeTensor antipode() const override;
    void fmexp(const FreeTensor &other) override;

};

namespace dtl {

template <typename Tensor>
Tensor exp_wrapper(const Tensor &arg) {
    return exp(arg);
}

template <typename Tensor>
Tensor log_wrapper(const Tensor &arg) {
    return log(arg);
}

template <typename Tensor>
Tensor inverse_wrapper(const Tensor &arg) {
    return inverse(arg);
}

template <typename Tensor>
Tensor antipode_wrapper(const Tensor &arg) {
    return antipode(arg);
}

}// namespace dtl

template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::exp() const {
    return FreeTensor(FreeTensorInterface::p_ctx, dtl::exp_wrapper(base_t::data()));
}
template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::log() const {
    return FreeTensor(FreeTensorInterface::p_ctx, dtl::log_wrapper(base_t::data()));
}
template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::inverse() const {
    return FreeTensor(FreeTensorInterface::p_ctx, dtl::inverse_wrapper(base_t::data()));
}
template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::antipode() const {
    return FreeTensor(FreeTensorInterface::p_ctx, dtl::antipode_wrapper(base_t::data()));
}
template <typename FTImpl, template <typename> class StorageModel>
void FreeTensorImplementation<FTImpl, StorageModel>::fmexp(const FreeTensor &other) {
    base_t::data().fmexp_inplace(FreeTensorImplementation::convert_argument(other));
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_
