// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_
#define ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_

#include "algebra_bundle_impl.h"
#include "algebra_impl.h"
#include "free_tensor.h"

namespace rpy {
namespace algebra {

template <typename FTImpl, template <typename> class StorageModel>
class FreeTensorImplementation
    : public AlgebraImplementation<FreeTensorInterface, FTImpl, StorageModel>
{
    using base_t
            = AlgebraImplementation<FreeTensorInterface, FTImpl, StorageModel>;

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

    RPY_NO_DISCARD FreeTensor exp() const override;
    RPY_NO_DISCARD FreeTensor log() const override;
//    RPY_NO_DISCARD FreeTensor inverse() const override;
    RPY_NO_DISCARD FreeTensor antipode() const override;
    void fmexp(const FreeTensor& other) override;
};

namespace dtl {

template <typename Tensor>
Tensor exp_wrapper(const Tensor& arg)
{
    return exp(arg);
}

template <typename Tensor>
Tensor log_wrapper(const Tensor& arg)
{
    return log(arg);
}

template <typename Tensor>
Tensor inverse_wrapper(const Tensor& arg)
{
    return inverse(arg);
}

template <typename Tensor>
Tensor antipode_wrapper(const Tensor& arg)
{
    return antipode(arg);
}

}// namespace dtl

template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::exp() const
{
    return FreeTensor(
            FreeTensorInterface::p_ctx, dtl::exp_wrapper(base_t::data())
    );
}
template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::log() const
{
    return FreeTensor(
            FreeTensorInterface::p_ctx, dtl::log_wrapper(base_t::data())
    );
}
//template <typename FTImpl, template <typename> class StorageModel>
//FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::inverse() const
//{
//    return FreeTensor(
//            FreeTensorInterface::p_ctx, dtl::inverse_wrapper(base_t::data())
//    );
//}
template <typename FTImpl, template <typename> class StorageModel>
FreeTensor FreeTensorImplementation<FTImpl, StorageModel>::antipode() const
{
    return FreeTensor(
            FreeTensorInterface::p_ctx, dtl::antipode_wrapper(base_t::data())
    );
}
template <typename FTImpl, template <typename> class StorageModel>
void FreeTensorImplementation<FTImpl, StorageModel>::fmexp(
        const FreeTensor& other
)
{
    base_t::data().fmexp_inplace(
            FreeTensorImplementation::convert_argument(other)
    );
}

template <typename FTBImpl, template <typename> class StorageModel>
class FreeTensorBundleImplementation
    : public AlgebraBundleImplementation<
              FreeTensorBundleInterface, FTBImpl, StorageModel>
{
    using base_t = AlgebraBundleImplementation<
            FreeTensorBundleInterface, FTBImpl, StorageModel>;

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
            FreeTensorBundleInterface::p_ctx, dtl::exp_wrapper(base_t::data())
    );
}
template <typename FTBImpl, template <typename> class StorageModel>
FreeTensorBundle
FreeTensorBundleImplementation<FTBImpl, StorageModel>::log() const
{
    return FreeTensorBundle(
            FreeTensorBundleInterface::p_ctx, dtl::log_wrapper(base_t::data())
    );
}
//template <typename FTBImpl, template <typename> class StorageModel>
//FreeTensorBundle
//FreeTensorBundleImplementation<FTBImpl, StorageModel>::inverse() const
//{
//    return FreeTensorBundle(
//            FreeTensorBundleInterface::p_ctx,
//            dtl::inverse_wrapper(base_t::data())
//    );
//}
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

#endif// ROUGHPY_ALGEBRA_FREE_TENSOR_IMPL_H_
