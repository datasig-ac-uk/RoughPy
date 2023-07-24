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

#ifndef ROUGHPY_LA_CONTEXT_LA_CONTEXT_H_
#define ROUGHPY_LA_CONTEXT_LA_CONTEXT_H_

#include "roughpy/algebra/algebra_base.h"
#include "roughpy/algebra/algebra_impl.h"
#include "roughpy/algebra/algebra_info.h"
#include "roughpy/algebra/algebra_iterator.h"
#include "roughpy/algebra/algebra_iterator_impl.h"
#include "roughpy/algebra/algebra_traits.h"
#include "roughpy/algebra/basis.h"
#include "roughpy/algebra/basis_impl.h"
#include "roughpy/algebra/context.h"
#include "roughpy/algebra/free_tensor.h"
#include "roughpy/algebra/free_tensor_impl.h"
#include "roughpy/algebra/lie.h"
#include "roughpy/algebra/shuffle_tensor.h"
#include "roughpy/scalars/key_scalar_array.h"
#include "roughpy/scalars/scalar.h"
#include "roughpy/scalars/scalar_array.h"
#include "roughpy/scalars/scalar_pointer.h"
#include "roughpy/scalars/scalar_stream.h"
#include "roughpy/scalars/scalar_type.h"

#include <libalgebra/libalgebra.h>

#include "la_context/free_tensor_info.h"
#include "la_context/lie_basis_info.h"
#include "la_context/lie_info.h"
#include "la_context/shuffle_tensor_info.h"
#include "la_context/tensor_basis_info.h"
#include "la_context/vector_iterator.h"
#include "la_context/vector_type_selector.h"

#include <libalgebra/utils.h>

namespace rpy {
namespace algebra {

namespace dtl {

template <AlgebraType ATYpe>
struct la_alg_type_tag {
};

}// namespace dtl

template <deg_t Width, deg_t Depth, typename Coefficients>
class LAContext : public Context
{
    LieBasis m_lie_basis;
    TensorBasis m_tensor_basis;

    template <VectorType VType>
    using lie_t =
            typename dtl::LAVectorSelector<VType>::template lie_t<Width, Depth,
                                                                  Coefficients>;

    template <VectorType VType>
    using free_tensor_t = typename dtl::LAVectorSelector<
            VType>::template ftensor_t<Width, Depth, Coefficients>;

    template <VectorType VType>
    using shuffle_tensor_t = typename dtl::LAVectorSelector<
            VType>::template stensor_t<Width, Depth, Coefficients>;

    template <VectorType VType>
    using maps_t = alg::maps<Coefficients, Width, Depth, free_tensor_t<VType>,
                             lie_t<VType>>;

    template <typename OutType, typename InType>
    OutType convert_impl(const InType& arg) const;

    template <typename OutType>
    OutType construct_impl(const VectorConstructionData& data) const;

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
    free_tensor_t<VType> Ad_x_n(deg_t n, const free_tensor_t<VType>& x,
                                const free_tensor_t<VType>& y) const;

    template <VectorType VType>
    free_tensor_t<VType>
    derive_series_compute(const free_tensor_t<VType>& increment,
                          const free_tensor_t<VType>& perturbation) const;

    template <VectorType VType>
    free_tensor_t<VType>
    sig_derivative_single(const free_tensor_t<VType>& signature,
                          const free_tensor_t<VType>& increment,
                          const free_tensor_t<VType>& perturbation) const;

    template <VectorType VType>
    free_tensor_t<VType>
    sig_derivative_impl(const std::vector<DerivativeComputeInfo>& info) const;

    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data,
                   dtl::la_alg_type_tag<AlgebraType::FreeTensor>) const;

    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data,
                   dtl::la_alg_type_tag<AlgebraType::ShuffleTensor>) const;

    UnspecifiedAlgebraType
    construct_impl(const VectorConstructionData& data,
                   dtl::la_alg_type_tag<AlgebraType::Lie>) const;

public:
    using scalar_type = typename Coefficients::SCA;
    using rational_type = typename Coefficients::RAT;

    LAContext();

    context_pointer get_alike(deg_t new_depth) const override;
    context_pointer
    get_alike(const scalars::ScalarType* new_ctype) const override;
    context_pointer
    get_alike(deg_t new_depth,
              const scalars::ScalarType* new_ctype) const override;
    context_pointer
    get_alike(deg_t new_width, deg_t new_depth,
              const scalars::ScalarType* new_ctype) const override;
    LieBasis get_lie_basis() const override;
    TensorBasis get_tensor_basis() const override;
    FreeTensor convert(const FreeTensor& arg,
                       optional<VectorType> new_vec_type) const override;
    ShuffleTensor convert(const ShuffleTensor& arg,
                          optional<VectorType> new_vec_type) const override;
    Lie convert(const Lie& arg,
                optional<VectorType> new_vec_type) const override;
    FreeTensor
    construct_free_tensor(const VectorConstructionData& arg) const override;
    ShuffleTensor
    construct_shuffle_tensor(const VectorConstructionData& arg) const override;
    Lie construct_lie(const VectorConstructionData& arg) const override;
    UnspecifiedAlgebraType
    construct(AlgebraType type,
              const VectorConstructionData& data) const override;
    FreeTensor lie_to_tensor(const Lie& arg) const override;
    Lie tensor_to_lie(const FreeTensor& arg) const override;
    FreeTensor signature(const SignatureData& data) const override;
    Lie log_signature(const SignatureData& data) const override;
    FreeTensor sig_derivative(const std::vector<DerivativeComputeInfo>& info,
                              VectorType vtype) const override;
};

template <deg_t Width, deg_t Depth, typename Coefficients>
template <typename OutType, typename InType>
OutType
LAContext<Width, Depth, Coefficients>::convert_impl(const InType& arg) const
{
    OutType result;

    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <typename OutType>
OutType LAContext<Width, Depth, Coefficients>::construct_impl(
        const VectorConstructionData& data) const
{
    OutType result;

    if (data.data.is_null()) { return result; }

    const auto& basis = OutType::basis;
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
            result[basis.index_to_key(keys[i])] = data_ptr[i];
        }

    } else {
        // Dense data

        for (dimn_t i = 0; i < size; ++i) {
            // Replace this with a more efficient method once it's implemented
            // at the lower level
            result[basis.index_to_key(i)] = data_ptr[i];
        }
    }

    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::convert_impl(const FreeTensor& arg) const
{
    free_tensor_t<VType> result;
    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::shuffle_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::convert_impl(
        const ShuffleTensor& arg) const
{
    shuffle_tensor_t<VType> result;
    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::lie_t<VType>
LAContext<Width, Depth, Coefficients>::convert_impl(const Lie& arg) const
{
    lie_t<VType> result;
    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::lie_to_tensor_impl(const Lie& arg) const
{

    const auto& arg_context = arg->context();
    if (arg_context == this) {
        maps_t<VType> maps;
        return maps.l2t(algebra_cast<lie_t<VType>>(*arg));
    }

    if (arg_context->width() != width()) {
        RPY_THROW(std::invalid_argument,
                "cannot perform conversion on algebras with different bases");
    }

    return convert_impl<VType>(arg_context->lie_to_tensor(arg));
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::lie_t<VType>
LAContext<Width, Depth, Coefficients>::tensor_to_lie_impl(
        const FreeTensor& arg) const
{
    const auto& arg_context = arg->context();

    maps_t<VType> maps;
    if (arg_context == this) {
        return maps.t2l(algebra_cast<free_tensor_t<VType>>(*arg));
    }

    if (arg_context->width() != width()) {
        RPY_THROW(std::invalid_argument,
                "cannot perform conversion on algebras with different bases");
    }

    return maps.t2l(convert_impl<VType>(arg));
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::lie_t<VType>
LAContext<Width, Depth, Coefficients>::cbh_impl(
        const std::vector<Lie>& lies) const
{
    maps_t<VType> maps;
    free_tensor_t<VType> collector;
    collector[typename free_tensor_t<VType>::KEY()] = scalar_type(1);
    for (const auto& lie : lies) {
        collector.fmexp_inplace(lie_to_tensor_impl<VType>(lie));
    }

    return maps.t2l(log(collector));
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::compute_signature(
        const SignatureData& data) const
{
    free_tensor_t<VType> result;
    result[typename free_tensor_t<VType>::KEY()] = scalar_type(1);
    const auto nrows = data.data_stream.row_count();
    maps_t<VType> maps;

    for (dimn_t i = 0; i < nrows; ++i) {
        auto row = data.data_stream[i];
        const auto* keys
                = data.key_stream.empty() ? nullptr : data.key_stream[i];
        VectorConstructionData row_cdata{scalars::KeyScalarArray(row, keys),
                                         VType};

        auto lie_row = construct_impl<lie_t<VType>>(row_cdata);

        // #if 0
        result.fmexp_inplace(maps.l2t(lie_row));
        // #endif
    }

    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::Ad_x_n(
        deg_t n, const LAContext::free_tensor_t<VType>& x,
        const LAContext::free_tensor_t<VType>& y) const
{
    auto tmp = x * y - y * x;
    while (--n) { tmp = x * tmp - tmp * x; }
    return tmp;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::derive_series_compute(
        const LAContext::free_tensor_t<VType>& increment,
        const LAContext::free_tensor_t<VType>& perturbation) const
{
    free_tensor_t<VType> result(perturbation);

    const auto depth = Depth;
    rational_type factor(1);
    for (deg_t d = 1; d <= depth; ++d) {
        factor *= rational_type(d + 1);
        if (d % 2 == 0) {
            result.add_scal_div(Ad_x_n<VType>(d, increment, perturbation),
                                factor);
        } else {
            result.sub_scal_div(Ad_x_n<VType>(d, increment, perturbation),
                                factor);
        }
    }

    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::sig_derivative_single(
        const LAContext::free_tensor_t<VType>& signature,
        const LAContext::free_tensor_t<VType>& increment,
        const LAContext::free_tensor_t<VType>& perturbation) const
{
    return signature * derive_series_compute<VType>(increment, perturbation);
}
template <deg_t Width, deg_t Depth, typename Coefficients>
template <VectorType VType>
LAContext<Width, Depth, Coefficients>::free_tensor_t<VType>
LAContext<Width, Depth, Coefficients>::sig_derivative_impl(
        const std::vector<DerivativeComputeInfo>& info) const
{
    free_tensor_t<VType> result;

    if (!info.empty()) {
        for (const auto& data : info) {
            auto tincr = lie_to_tensor_impl<VType>(data.logsig_of_interval);
            auto tperturb = lie_to_tensor_impl<VType>(data.perturbation);
            auto sig = exp(tincr);

            result *= sig;
            result += sig_derivative_single<VType>(sig, tincr, tperturb);
        }
    }

    return result;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
UnspecifiedAlgebraType LAContext<Width, Depth, Coefficients>::construct_impl(
        const VectorConstructionData& data,
        dtl::la_alg_type_tag<AlgebraType::FreeTensor>) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    new FreeTensorImplementation<free_tensor_t<(VTYPE)>, OwnedStorageModel>(   \
            this, construct_impl<free_tensor_t<(VTYPE)>>(data))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
UnspecifiedAlgebraType LAContext<Width, Depth, Coefficients>::construct_impl(
        const VectorConstructionData& data,
        dtl::la_alg_type_tag<AlgebraType::ShuffleTensor>) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    new AlgebraImplementation<ShuffleTensorInterface,                          \
                              shuffle_tensor_t<(VTYPE)>, OwnedStorageModel>(   \
            this, construct_impl<shuffle_tensor_t<(VTYPE)>>(data))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
UnspecifiedAlgebraType LAContext<Width, Depth, Coefficients>::construct_impl(
        const VectorConstructionData& data,
        dtl::la_alg_type_tag<AlgebraType::Lie>) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    new AlgebraImplementation<LieInterface, lie_t<(VTYPE)>,                    \
                              OwnedStorageModel>(                              \
            this, construct_impl<lie_t<(VTYPE)>>(data))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
LAContext<Width, Depth, Coefficients>::LAContext()
    : Context(Width, Depth, scalars::ScalarType::of<scalar_type>(),
              std::string("libalgebra")),
      m_tensor_basis(std::addressof(free_tensor_t<VectorType::Dense>::basis)),
      m_lie_basis(std::addressof(lie_t<VectorType::Dense>::basis))
{}

template <deg_t Width, deg_t Depth, typename Coefficients>
context_pointer
LAContext<Width, Depth, Coefficients>::get_alike(deg_t new_depth) const
{
    return get_context(Width, new_depth, ctype(), {{"backend", "libalgebra"}});
}
template <deg_t Width, deg_t Depth, typename Coefficients>
context_pointer LAContext<Width, Depth, Coefficients>::get_alike(
        const scalars::ScalarType* new_ctype) const
{
    return get_context(Width, Depth, new_ctype, {{"backend", "libalgebra"}});
}
template <deg_t Width, deg_t Depth, typename Coefficients>
context_pointer LAContext<Width, Depth, Coefficients>::get_alike(
        deg_t new_depth, const scalars::ScalarType* new_ctype) const
{
    return get_context(Width, new_depth, new_ctype,
                       {{"backend", "libalgebra"}});
}
template <deg_t Width, deg_t Depth, typename Coefficients>
context_pointer LAContext<Width, Depth, Coefficients>::get_alike(
        deg_t new_width, deg_t new_depth,
        const scalars::ScalarType* new_ctype) const
{
    return get_context(new_width, new_depth, new_ctype,
                       {{"backend", "libalgebra"}});
}
template <deg_t Width, deg_t Depth, typename Coefficients>
LieBasis LAContext<Width, Depth, Coefficients>::get_lie_basis() const
{
    return m_lie_basis;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
TensorBasis LAContext<Width, Depth, Coefficients>::get_tensor_basis() const
{
    return m_tensor_basis;
}
template <deg_t Width, deg_t Depth, typename Coefficients>
FreeTensor LAContext<Width, Depth, Coefficients>::convert(
        const FreeTensor& arg, optional<VectorType> new_vec_type) const
{
    auto vtype = (new_vec_type.has_value()) ? new_vec_type.value()
                                            : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
ShuffleTensor LAContext<Width, Depth, Coefficients>::convert(
        const ShuffleTensor& arg, optional<VectorType> new_vec_type) const
{
    auto vtype = (new_vec_type.has_value()) ? new_vec_type.value()
                                            : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) ShuffleTensor(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
Lie LAContext<Width, Depth, Coefficients>::convert(
        const Lie& arg, optional<VectorType> new_vec_type) const
{
    auto vtype = (new_vec_type.has_value()) ? new_vec_type.value()
                                            : arg.storage_type();
#define RPY_SWITCH_FN(VTYPE) Lie(this, convert_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
FreeTensor LAContext<Width, Depth, Coefficients>::construct_free_tensor(
        const VectorConstructionData& arg) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    FreeTensor(this, construct_impl<free_tensor_t<(VTYPE)>>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
ShuffleTensor LAContext<Width, Depth, Coefficients>::construct_shuffle_tensor(
        const VectorConstructionData& arg) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    ShuffleTensor(this, construct_impl<shuffle_tensor_t<(VTYPE)>>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
Lie LAContext<Width, Depth, Coefficients>::construct_lie(
        const VectorConstructionData& arg) const
{
#define RPY_SWITCH_FN(VTYPE) Lie(this, construct_impl<lie_t<(VTYPE)>>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
UnspecifiedAlgebraType LAContext<Width, Depth, Coefficients>::construct(
        AlgebraType type, const VectorConstructionData& data) const
{
#define RPY_SWITCH_FN(ATYPE)                                                   \
    construct_impl(data, dtl::la_alg_type_tag<(ATYPE)>())
    RPY_MAKE_ALGTYPE_SWITCH(type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
FreeTensor
LAContext<Width, Depth, Coefficients>::lie_to_tensor(const Lie& arg) const
{
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, lie_to_tensor_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
Lie LAContext<Width, Depth, Coefficients>::tensor_to_lie(
        const FreeTensor& arg) const
{
#define RPY_SWITCH_FN(VTYPE) Lie(this, tensor_to_lie_impl<(VTYPE)>(arg))
    RPY_MAKE_VTYPE_SWITCH(arg.storage_type())
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
FreeTensor LAContext<Width, Depth, Coefficients>::signature(
        const SignatureData& data) const
{
#define RPY_SWITCH_FN(VTYPE) FreeTensor(this, compute_signature<(VTYPE)>(data))
    RPY_MAKE_VTYPE_SWITCH(data.vector_type)
#undef RPY_SWITCH_FN
}
template <deg_t Width, deg_t Depth, typename Coefficients>
Lie LAContext<Width, Depth, Coefficients>::log_signature(
        const SignatureData& data) const
{
    return tensor_to_lie(signature(data).log());
}
template <deg_t Width, deg_t Depth, typename Coefficients>
FreeTensor LAContext<Width, Depth, Coefficients>::sig_derivative(
        const std::vector<DerivativeComputeInfo>& info, VectorType vtype) const
{
#define RPY_SWITCH_FN(VTYPE)                                                   \
    FreeTensor(this, sig_derivative_impl<(VTYPE)>(info))
    RPY_MAKE_VTYPE_SWITCH(vtype)
#undef RPY_SWITCH_FN
}

}// namespace algebra
}// namespace rpy

#endif// ROUGHPY_LA_CONTEXT_LA_CONTEXT_H_
