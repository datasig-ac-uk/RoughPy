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

//
// Created by user on 06/03/23.
//

#ifndef ROUGHPY_ALGEBRA_SRC_LITE_CONTEXT_H
#define ROUGHPY_ALGEBRA_SRC_LITE_CONTEXT_H

#include <roughpy/scalars/key_scalar_array.h>
#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_pointer.h>
#include <roughpy/scalars/scalar_stream.h>
#include <roughpy/scalars/scalar_type.h>

#include <roughpy/algebra/algebra_base.h>
#include <roughpy/algebra/algebra_impl.h>
#include <roughpy/algebra/algebra_info.h>
#include <roughpy/algebra/algebra_iterator.h>
#include <roughpy/algebra/algebra_iterator_impl.h>
#include <roughpy/algebra/basis.h>
#include <roughpy/algebra/basis_impl.h>
#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/free_tensor_impl.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/shuffle_tensor.h>

#include "libalgebra_lite_internal/algebra_type_caster.h"
#include "libalgebra_lite_internal/algebra_type_helper.h"
#include "libalgebra_lite_internal/dense_vector_iterator.h"
#include "libalgebra_lite_internal/free_tensor_info.h"
#include "libalgebra_lite_internal/lie_basis_info.h"
#include "libalgebra_lite_internal/lie_info.h"
#include "libalgebra_lite_internal/lite_vector_selector.h"
#include "libalgebra_lite_internal/shuffle_tensor_info.h"
#include "libalgebra_lite_internal/sparse_mutable_ref_scalar_trait.h"
#include "libalgebra_lite_internal/sparse_vector_iterator.h"
#include "libalgebra_lite_internal/tensor_basis_info.h"
#include "libalgebra_lite_internal/unspecified_algebra_binary_op.h"

#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/maps.h>
#include <libalgebra_lite/registry.h>

namespace rpy {
namespace algebra {

namespace dtl {

class LiteContextBasisHolder
{
protected:
    lal::basis_pointer<lal::tensor_basis> p_tbasis;
    lal::basis_pointer<lal::hall_basis> p_lbasis;

    LiteContextBasisHolder(deg_t width, deg_t depth)
        : p_tbasis(lal::basis_registry<lal::tensor_basis>::get(
                lal::deg_t(width), lal::deg_t(depth)
        )),
          p_lbasis(lal::basis_registry<lal::hall_basis>::get(
                  lal::deg_t(width), lal::deg_t(depth)
          ))
    {}
};

}// namespace dtl

template <typename Coefficients>
class LiteContext : private dtl::LiteContextBasisHolder, public Context
{
    TensorBasis m_tensor_basis;
    LieBasis m_lie_basis;

    using dtl::LiteContextBasisHolder::p_lbasis;
    using dtl::LiteContextBasisHolder::p_tbasis;

    std::shared_ptr<const lal::free_tensor_multiplication> p_ftmul;
    std::shared_ptr<const lal::lie_multiplication> p_liemul;
    std::shared_ptr<const lal::shuffle_tensor_multiplication> p_stmul;

    using coeff_traits = lal::coefficient_trait<Coefficients>;

public:
    using scalar_type = typename coeff_traits::scalar_type;
    using rational_type = typename coeff_traits::rational_type;

private:
    lal::maps m_maps;

    template <AlgebraType AType, VectorType VType>
    using arg_type_caster = algebra_type_caster<
            Coefficients, algebra_type_tag<AType>, vector_type_tag<VType>>;

    using binary_invoker = UnspecifiedFunctionInvoker<
            arg_type_caster, LiteContext, ConstRawUnspecifiedAlgebraType,
            ConstRawUnspecifiedAlgebraType>;

    template <typename T>
    RPY_NO_UBSAN static const T&
    unspecified_cast(ConstRawUnspecifiedAlgebraType arg)
    {
        using info = dtl::alg_details_of<T>;
        RPY_CHECK(arg != nullptr);
        RPY_CHECK(arg->alg_type() == info::alg_type);
        RPY_CHECK(arg->storage_type() == info::vec_type);
        const auto* interface_ptr
                = reinterpret_cast<const typename info::interface_type*>(arg);

        return algebra_cast<T, typename info::interface_type>(*interface_ptr);
    }

    template <typename OutType, typename InType>
    OutType convert_impl(
            const InType& arg,
            const lal::basis_pointer<typename OutType::basis_type>& basis,
            const std::shared_ptr<const typename OutType::multiplication_type>&
                    mul
    ) const;

    template <typename OutType>
    OutType construct_impl(
            const VectorConstructionData& data,
            const lal::basis_pointer<const typename OutType::basis_type>& basis,
            const std::shared_ptr<const typename OutType::multiplication_type>&
                    mul
    ) const;

    template <VectorType VType>
    using free_tensor_t = typename dtl::vector_type_selector<
            VType>::template free_tensor<Coefficients>;

    template <VectorType VType>
    using shuffle_tensor_t = typename dtl::vector_type_selector<
            VType>::template shuffle_tensor<Coefficients>;

    template <VectorType VType>
    using lie_t = typename dtl::vector_type_selector<VType>::template lie<
            Coefficients>;

    template <VectorType VType>
    free_tensor_t<VType> convert_impl(const FreeTensor& arg) const;
    template <VectorType VType>
    shuffle_tensor_t<VType> convert_impl(const ShuffleTensor& arg) const;
    template <VectorType VType>
    lie_t<VType> convert_impl(const Lie& arg) const;

    template <VectorType VType>
    free_tensor_t<VType> lie_to_tensor_impl(const Lie& arg) const;
    template <VectorType VType>
    lie_t<VType> tensor_to_lie_impl(const FreeTensor& arg) const;
    template <VectorType VType>
    lie_t<VType> cbh_impl(const std::vector<Lie>& lies) const;
    template <VectorType VType>
    free_tensor_t<VType> compute_signature(const SignatureData& data) const;
    template <VectorType VType>
    free_tensor_t<VType>
    Ad_x_n(deg_t d, const free_tensor_t<VType>& x,
           const free_tensor_t<VType>& y) const;
    template <VectorType VType>
    free_tensor_t<VType> derive_series_compute(
            const free_tensor_t<VType>& increment,
            const free_tensor_t<VType>& t_perturbation
    ) const;
    template <VectorType VType>
    free_tensor_t<VType> sig_derivative_single(
            const free_tensor_t<VType>& signature,
            const free_tensor_t<VType>& t_incr,
            const free_tensor_t<VType>& perturbation
    ) const;
    template <VectorType VType>
    free_tensor_t<VType>
    sig_derivative_impl(const std::vector<DerivativeComputeInfo>& info) const;

    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::FreeTensor>)
            const;
    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::ShuffleTensor>)
            const;
    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::Lie>)
            const;
    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::FreeTensorBundle>)
            const;
    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::ShuffleTensorBundle>)
            const;
    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::LieBundle>)
            const;

public:
    explicit LiteContext(deg_t width, deg_t depth);

    context_pointer get_alike(deg_t new_depth) const override;
    context_pointer get_alike(const scalars::ScalarType* new_ctype
    ) const override;
    context_pointer get_alike(
            deg_t new_depth, const scalars::ScalarType* new_ctype
    ) const override;
    context_pointer get_alike(
            deg_t new_width, deg_t new_depth,
            const scalars::ScalarType* new_ctype
    ) const override;
    LieBasis get_lie_basis() const override;
    TensorBasis get_tensor_basis() const override;
    FreeTensor
    convert(const FreeTensor& arg,
            optional<VectorType> new_vec_type) const override;
    ShuffleTensor
    convert(const ShuffleTensor& arg,
            optional<VectorType> new_vec_type) const override;
    Lie
    convert(const Lie& arg, optional<VectorType> new_vec_type) const override;
    FreeTensor construct_free_tensor(const VectorConstructionData& arg
    ) const override;
    ShuffleTensor construct_shuffle_tensor(const VectorConstructionData& arg
    ) const override;
    Lie construct_lie(const VectorConstructionData& arg) const override;
    UnspecifiedAlgebraType construct(
            AlgebraType type, const VectorConstructionData& data
    ) const override;

    FreeTensor lie_to_tensor(const Lie& arg) const override;
    Lie tensor_to_lie(const FreeTensor& arg) const override;
    FreeTensor signature(const SignatureData& data) const override;
    Lie log_signature(const SignatureData& data) const override;
    FreeTensor sig_derivative(
            const std::vector<DerivativeComputeInfo>& info, VectorType vtype
    ) const override;

private:
    template <typename Left, typename Right, typename FreeFunctionWrapper>
    UnspecifiedAlgebraType algebra_free_multiply_func_impl2(
            const Left& left, const Right& right, FreeFunctionWrapper&& wrapper
    ) const
    {
        using OutType = decltype(wrapper(left, right));
        using impl_t =
                typename dtl::alg_details_of<OutType>::implementation_type;
        return UnspecifiedAlgebraType(new impl_t(this, wrapper(left, right)));
    }

public:
    UnspecifiedAlgebraType free_multiply(
            ConstRawUnspecifiedAlgebraType left,
            ConstRawUnspecifiedAlgebraType right
    ) const override;
    UnspecifiedAlgebraType shuffle_multiply(
            ConstRawUnspecifiedAlgebraType left,
            ConstRawUnspecifiedAlgebraType right
    ) const override;
    UnspecifiedAlgebraType half_shuffle_multiply(
            ConstRawUnspecifiedAlgebraType left,
            ConstRawUnspecifiedAlgebraType right
    ) const override;
    UnspecifiedAlgebraType adjoint_to_left_multiply_by(
            ConstRawUnspecifiedAlgebraType multiplier,
            ConstRawUnspecifiedAlgebraType argument
    ) const override;
};

class LiteContextMaker : public ContextMaker
{
    using ContextMaker::preference_list;
    context_pointer create_context(
            deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            const preference_list& preferences
    ) const;

public:
    bool
    can_get(deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            const preference_list& preferences) const override;
    context_pointer get_context(
            deg_t width, deg_t depth, const scalars::ScalarType* ctype,
            const preference_list& preferences
    ) const override;
    optional<base_context_pointer>
    get_base_context(deg_t width, deg_t depth) const override;
};

template <typename Coefficients>
template <typename OutType, typename InType>
OutType LiteContext<Coefficients>::convert_impl(
        const InType& arg,
        const lal::basis_pointer<typename OutType::basis_type>& basis,
        const std::shared_ptr<const typename OutType::multiplication_type>& mul
) const
{
    OutType result(basis, mul);

    // TODO: Needs implementation
    if (arg->context() == this) {}

    /*
     * If arg is not an object from this context then there are some options:
     *    1) It is from a libalgebra_lite context with different
     *       width/depth/scalar combinations;
     *    2) It is from a compatible context (e.g. libalgebra) (possibly with
     *       different width/depth/scalar combinations);
     *    3) It is from some other context that is not compatible.
     */

    return result;
}

namespace dtl {

template <typename TensorObject>
void tensor_populate_vcd(VectorConstructionData& data, const TensorObject& arg)
{

    if (arg.storage_type() == VectorType::Dense) {
        // Simply borrow the data pointer
        data.data = *arg.dense_data();
    } else {
        // Construct key-value arrays to pass to construct_impl
        auto sz = arg.size();
        data.data.allocate_scalars(sz);
        data.data.allocate_keys();

        auto* key_array = data.data.keys();

        dimn_t i = 0;
        for (auto&& it : arg) {
            data.data[i] = it->value();
            key_array[i] = it->key();
            ++i;
        }
        RPY_DBG_ASSERT(i == sz);
    }
}

}// namespace dtl

template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::convert_impl(const FreeTensor& arg) const
{
    /*
     * Tensor bases are assumed to be order-isomorphic to one another.
     * So it is quite safe to copy the data across from the source tensor
     * in whatever mode is most appropriate. We'll do this by making a
     * VectorConstructionData object and then passing this to the
     * construct_impl method
     */
    VectorConstructionData data{
            scalars::KeyScalarArray(arg.coeff_type()),
            VType// Not that this really matters;
    };
    dtl::tensor_populate_vcd(data, arg);
    return construct_impl<free_tensor_t<VType>>(data, p_tbasis, p_ftmul);
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template shuffle_tensor_t<VType>
LiteContext<Coefficients>::convert_impl(const ShuffleTensor& arg) const
{
    /*
     * See comments in the convert_impl for FreeTensor. The same applies here.
     */
    VectorConstructionData data{
            scalars::KeyScalarArray(arg.coeff_type()),
            VType// Not that this really matters;
    };
    dtl::tensor_populate_vcd(data, arg);
    return construct_impl<shuffle_tensor_t<VType>>(data, p_tbasis, p_stmul);
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template lie_t<VType>
LiteContext<Coefficients>::convert_impl(const Lie& arg) const
{
    /*
     * Lie bases need not be order-isomorphic to one another. We get around this
     * by factoring the Lie object through the FreeTensor. So we use
     * arg.context() to convert lie_to_tensor, convert the tensor using the
     * above implementation, and then use tensor_to_lie (from this) to convert
     * back to a lie_t<VType>.
     */
    auto tensor_version = arg->context()->lie_to_tensor(arg);
    return m_maps.tensor_to_lie(convert_impl<VType>(tensor_version));
}

template <typename Coefficients>
template <typename OutType>
OutType LiteContext<Coefficients>::construct_impl(
        const VectorConstructionData& data,
        const lal::basis_pointer<const typename OutType::basis_type>& basis,
        const std::shared_ptr<const typename OutType::multiplication_type>& mul
) const
{
    OutType result(basis, mul);

    if (data.data.is_null()) { return result; }

    const scalar_type* data_ptr;

    const auto size = data.data.size();
    std::vector<scalar_type> tmp;
    if (data.data.type() != ctype()) {
        tmp.resize(data.data.size());
        ctype()->convert_copy(tmp.data(), data.data, size);
        data_ptr = tmp.data();
    } else {
        data_ptr = data.data.raw_cast<const scalar_type>();
    }

    if (data.data.has_keys()) {
        // Sparse data
        const auto* keys = data.data.keys();

        for (dimn_t i = 0; i < size; ++i) {
            result[basis->index_to_key(keys[i])] = data_ptr[i];
        }

    } else {
        // Dense data

        for (dimn_t i = 0; i < size; ++i) {
            // Replace this with a more efficient method once it's implemented
            // at the lower level
            result[basis->index_to_key(i)] = data_ptr[i];
        }
    }

    return result;
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::lie_to_tensor_impl(const Lie& arg) const
{
    // If arg is an object from this context, then we can just use the maps
    // directly
    const auto& arg_context = arg->context();
    if (arg_context == this) {
        return m_maps.lie_to_tensor(algebra_cast<lie_t<VType>>(*arg));
    }

    if (arg_context->width() != width()) {
        RPY_THROW(std::invalid_argument,
                "cannot perform conversion on algebras with different bases"
        );
    }

    return convert_impl<VType>(arg_context->lie_to_tensor(arg));
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template lie_t<VType>
LiteContext<Coefficients>::tensor_to_lie_impl(const FreeTensor& arg) const
{

    const auto& arg_context = arg->context();
    if (arg_context == this) {
        return m_maps.tensor_to_lie(algebra_cast<free_tensor_t<VType>>(*arg));
    }

    if (arg_context->width() != width()) {
        RPY_THROW(std::invalid_argument,
                "cannot perform conversion on algebras with different bases"
        );
    }

    return m_maps.tensor_to_lie(convert_impl<VType>(arg));
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template lie_t<VType>
LiteContext<Coefficients>::cbh_impl(const std::vector<Lie>& lies) const
{

    free_tensor_t<VType> collector(p_tbasis, p_ftmul);
    collector[typename lal::tensor_basis::key_type()] = scalar_type(1);
    for (const auto& lie : lies) {
        collector.fmexp_inplace(lie_to_tensor_impl<VType>(lie));
    }

    return m_maps.tensor_to_lie(log(collector));
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::compute_signature(const SignatureData& data) const
{
    free_tensor_t<VType> result(p_tbasis, p_ftmul);
    result[typename lal::tensor_basis::key_type()] = scalar_type(1);
    const auto nrows = data.data_stream.row_count();

    for (dimn_t i = 0; i < nrows; ++i) {
        auto row = data.data_stream[i];
        const auto* keys
                = data.key_stream.empty() ? nullptr : data.key_stream[i];
        VectorConstructionData row_cdata{
                scalars::KeyScalarArray(row, keys), VType};

        auto lie_row
                = construct_impl<lie_t<VType>>(row_cdata, p_lbasis, p_liemul);

        // #if 0
        result.fmexp_inplace(m_maps.lie_to_tensor(lie_row));
        // #endif
    }

    return result;
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::Ad_x_n(
        deg_t d, const LiteContext::free_tensor_t<VType>& x,
        const LiteContext::free_tensor_t<VType>& y
) const
{
    auto tmp = x * y - y * x;
    while (--d) { tmp = x * tmp - tmp * x; }
    return tmp;
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::derive_series_compute(
        const LiteContext::free_tensor_t<VType>& increment,
        const LiteContext::free_tensor_t<VType>& t_perturbation
) const
{
    free_tensor_t<VType> result(t_perturbation);

    auto depth = ContextBase::depth();
    typename Coefficients::rational_type factor(1);

    auto ad_x = commutator(increment, t_perturbation);
    for (deg_t d = 1; d <= depth; ++d) {
        factor *= typename Coefficients::rational_type(d + 1);
        if (d % 2 == 0) {
            result.add_scal_div(ad_x, factor);
        } else {
            result.sub_scal_div(ad_x, factor);
        }
        ad_x = commutator(increment, ad_x);
    }
    return result;
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::sig_derivative_single(
        const LiteContext::free_tensor_t<VType>& signature,
        const LiteContext::free_tensor_t<VType>& t_incr,
        const LiteContext::free_tensor_t<VType>& perturbation
) const
{
    return signature * derive_series_compute<VType>(t_incr, perturbation);
}
template <typename Coefficients>
template <VectorType VType>
typename LiteContext<Coefficients>::template free_tensor_t<VType>
LiteContext<Coefficients>::sig_derivative_impl(
        const std::vector<DerivativeComputeInfo>& info
) const
{
    using tensor_type = free_tensor_t<VType>;

    if (info.empty()) { return tensor_type(p_tbasis, p_ftmul); }

    tensor_type result(p_tbasis, p_ftmul);

    for (const auto& data : info) {
        auto tincr = lie_to_tensor_impl<VType>(data.logsig_of_interval);
        auto tperturb = lie_to_tensor_impl<VType>(data.perturbation);
        auto signature = exp(tincr);

        result *= signature;
        result += sig_derivative_single<VType>(signature, tincr, tperturb);
    }

    return result;
}

template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::FreeTensor>)
                const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    UnspecifiedAlgebraType(new FreeTensorImplementation<                       \
                           free_tensor_t<(VTYPE)>, OwnedStorageModel>(         \
            this,                                                              \
            construct_impl<free_tensor_t<(VTYPE)>>(data, p_tbasis, p_ftmul)    \
    ))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::ShuffleTensor>)
                const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    UnspecifiedAlgebraType(new AlgebraImplementation<                          \
                           ShuffleTensorInterface, shuffle_tensor_t<(VTYPE)>,  \
                           OwnedStorageModel>(                                 \
            this,                                                              \
            construct_impl<shuffle_tensor_t<(VTYPE)>>(data, p_tbasis, p_stmul) \
    ))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::Lie>)
                const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    UnspecifiedAlgebraType(new AlgebraImplementation<                          \
                           LieInterface, lie_t<(VTYPE)>, OwnedStorageModel>(   \
            this, construct_impl<lie_t<(VTYPE)>>(data, p_lbasis, p_liemul)     \
    ))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::FreeTensorBundle>)
                const
{
    RPY_THROW(std::runtime_error, "not implemented for FreeTensorBundle");
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::ShuffleTensorBundle>)
                const
{
    RPY_THROW(std::runtime_error, "not implemented for ShuffleTensorBundle");
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::
        construct_impl(const VectorConstructionData& data, algebra_type_tag<AlgebraType::LieBundle>)
                const
{
    RPY_THROW(std::runtime_error, "not implemented for LieBundle");
}

template <typename Coefficients>
LiteContext<Coefficients>::LiteContext(deg_t width, deg_t depth)
    : dtl::LiteContextBasisHolder(width, depth),
      Context(width, depth, scalars::ScalarType::of<scalar_type>(),
              string("libalgebra_lite"), p_lbasis->sizes().data(),
              p_tbasis->sizes().data()),
      m_tensor_basis(&*p_tbasis), m_lie_basis(&*p_lbasis),
      p_ftmul(lal::multiplication_registry<
              lal::free_tensor_multiplication>::get(*p_tbasis)),
      p_liemul(lal::multiplication_registry<lal::lie_multiplication>::get(
              *p_lbasis
      )),
      p_stmul(lal::multiplication_registry<
              lal::shuffle_tensor_multiplication>::get(*p_tbasis)),
      m_maps(p_tbasis, p_lbasis)
{}

template <typename Coefficients>
context_pointer LiteContext<Coefficients>::get_alike(deg_t new_depth) const
{
    return get_context(
            width(), new_depth, ctype(),
            {
                    {"backend", "libalgebra_lite"}
    }
    );
}
template <typename Coefficients>
context_pointer
LiteContext<Coefficients>::get_alike(const scalars::ScalarType* new_ctype) const
{
    return get_context(
            width(), depth(), new_ctype,
            {
                    {"backend", "libalgebra_lite"}
    }
    );
}
template <typename Coefficients>
context_pointer LiteContext<Coefficients>::get_alike(
        deg_t new_depth, const scalars::ScalarType* new_ctype
) const
{
    return get_context(
            width(), new_depth, new_ctype,
            {
                    {"backend", "libalgebra_lite"}
    }
    );
}
template <typename Coefficients>
context_pointer LiteContext<Coefficients>::get_alike(
        deg_t new_width, deg_t new_depth, const scalars::ScalarType* new_ctype
) const
{
    return get_context(
            new_width, new_depth, new_ctype,
            {
                    {"backend", "libalgebra_lite"}
    }
    );
}
template <typename Coefficients>
LieBasis LiteContext<Coefficients>::get_lie_basis() const
{
    return m_lie_basis;
}
template <typename Coefficients>
TensorBasis LiteContext<Coefficients>::get_tensor_basis() const
{
    return m_tensor_basis;
}

template <typename Coefficients>
FreeTensor LiteContext<Coefficients>::convert(
        const FreeTensor& arg, optional<VectorType> new_vec_type
) const
{
    auto vtype
            = (new_vec_type.has_value()) ? *new_vec_type : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
ShuffleTensor LiteContext<Coefficients>::convert(
        const ShuffleTensor& arg, optional<VectorType> new_vec_type
) const
{
    auto vtype
            = (new_vec_type.has_value()) ? *new_vec_type : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) ShuffleTensor(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
Lie LiteContext<Coefficients>::convert(
        const Lie& arg, optional<VectorType> new_vec_type
) const
{
    auto vtype
            = (new_vec_type.has_value()) ? *new_vec_type : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) Lie(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
FreeTensor LiteContext<Coefficients>::construct_free_tensor(
        const VectorConstructionData& arg
) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    FreeTensor(                                                                \
            this,                                                              \
            construct_impl<free_tensor_t<(VTYPE)>>(arg, p_tbasis, p_ftmul)     \
    )
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
ShuffleTensor LiteContext<Coefficients>::construct_shuffle_tensor(
        const VectorConstructionData& arg
) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    ShuffleTensor(                                                             \
            this,                                                              \
            construct_impl<shuffle_tensor_t<(VTYPE)>>(arg, p_tbasis, p_stmul)  \
    )
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
Lie LiteContext<Coefficients>::construct_lie(const VectorConstructionData& arg
) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    Lie(this, construct_impl<lie_t<(VTYPE)>>(arg, p_lbasis, p_liemul))
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::construct(
        AlgebraType type, const VectorConstructionData& data
) const
{
#define RPY_SWITCH_FN(ATYPE) construct_impl(data, algebra_type_tag<ATYPE>())
    RPY_MAKE_ALGTYPE_SWITCH(type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
FreeTensor LiteContext<Coefficients>::lie_to_tensor(const Lie& arg) const
{
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, lie_to_tensor_impl<VTYPE>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef RPY_SWITCH_FN
}

template <typename Coefficients>
Lie LiteContext<Coefficients>::tensor_to_lie(const FreeTensor& arg) const
{
#define RPY_SWITCH_FN(VTYPE) Lie(this, tensor_to_lie_impl<VTYPE>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
FreeTensor LiteContext<Coefficients>::signature(const SignatureData& data) const
{
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, compute_signature<VTYPE>(data))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <typename Coefficients>
Lie LiteContext<Coefficients>::log_signature(const SignatureData& data) const
{
    return tensor_to_lie(signature(data).log());
}
template <typename Coefficients>
FreeTensor LiteContext<Coefficients>::sig_derivative(
        const std::vector<DerivativeComputeInfo>& info, VectorType vtype
) const
{
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, sig_derivative_impl<VTYPE>(info))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}

namespace wrappers {

struct FreeMultiply {
    template <typename L, typename R>
    auto operator()(const L& left, const R& right) const
            -> decltype(lal::free_tensor_multiply(left, right))
    {
        return lal::free_tensor_multiply(left, right);
    }
    template <AlgebraType Type>
    using compatible = integral_constant<
            bool,
            Type == AlgebraType::FreeTensor
                    || Type == AlgebraType::ShuffleTensor>;
};
struct ShuffleMultiply {

    template <typename L, typename R>
    auto operator()(const L& left, const R& right) const
            -> decltype(lal::shuffle_multiply(left, right))
    {
        return lal::shuffle_multiply(left, right);
    }
    template <AlgebraType Type>
    using compatible = integral_constant<
            bool,
            Type == AlgebraType::FreeTensor
                    || Type == AlgebraType::ShuffleTensor>;
};

struct HalfShuffleMultiply {

    template <typename L, typename R>
    auto operator()(const L& left, const R& right) const
            -> decltype(lal::half_shuffle_multiply(left, right))
    {
        return lal::half_shuffle_multiply(left, right);
    }
    template <AlgebraType Type>
    using compatible = integral_constant<
            bool,
            Type == AlgebraType::FreeTensor
                    || Type == AlgebraType::ShuffleTensor>;
};

struct AdjointFreeMultiply {

    template <typename M, typename A>
    auto operator()(const M& multiplier, const A& arg) const
            -> decltype(lal::left_free_tensor_multiply_adjoint(multiplier, arg))
    {
        return lal::left_free_tensor_multiply_adjoint(multiplier, arg);
    }

    template <AlgebraType Type>
    using compatible = integral_constant<
            bool,
            Type == AlgebraType::FreeTensor
                    || Type == AlgebraType::ShuffleTensor>;
};

}// namespace wrappers

template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::free_multiply(
        ConstRawUnspecifiedAlgebraType left,
        ConstRawUnspecifiedAlgebraType right
) const
{
    return binary_invoker::eval(
            this, wrappers::FreeMultiply(), move(left), move(right)
    );
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::shuffle_multiply(
        ConstRawUnspecifiedAlgebraType left,
        ConstRawUnspecifiedAlgebraType right
) const
{
    return binary_invoker::eval(
            this, wrappers::ShuffleMultiply(), move(left), move(right)
    );
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::half_shuffle_multiply(
        ConstRawUnspecifiedAlgebraType left,
        ConstRawUnspecifiedAlgebraType right
) const
{
    return binary_invoker::eval(
            this, wrappers::HalfShuffleMultiply(), move(left), move(right)
    );
}
template <typename Coefficients>
UnspecifiedAlgebraType LiteContext<Coefficients>::adjoint_to_left_multiply_by(
        ConstRawUnspecifiedAlgebraType multiplier,
        ConstRawUnspecifiedAlgebraType argument
) const
{
    return binary_invoker::eval(
            this, wrappers::AdjointFreeMultiply(), move(multiplier),
            move(argument)
    );
}

extern template class LiteContext<lal::float_field>;

extern template class LiteContext<lal::double_field>;

extern template class LiteContext<lal::rational_field>;

extern template class LiteContext<lal::polynomial_ring>;

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_ALGEBRA_SRC_LITE_CONTEXT_H
