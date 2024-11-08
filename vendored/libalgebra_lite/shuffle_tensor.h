// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 31/01/23.
//

#ifndef LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_SHUFFLE_TENSOR_H
#define LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_SHUFFLE_TENSOR_H

#include "implementation_types.h"
#include "libalgebra_lite_export.h"

#include "algebra.h"
#include "basis_traits.h"
#include "coefficients.h"
#include "free_tensor.h"
#include "tensor_basis.h"

namespace lal {

class left_half_shuffle_tensor_multiplier;

class right_half_shuffle_tensor_multiplier;

LAL_EXPORT_TEMPLATE_CLASS(
        base_multiplier, left_half_shuffle_tensor_multiplier, tensor_basis
)
LAL_EXPORT_TEMPLATE_CLASS(
        base_multiplier, right_half_shuffle_tensor_multiplier, tensor_basis
)

#if 0
class free_tensor_multiplication
{



    template <typename Coefficients, typename Fn>
    void fma_dense_giles(dtl::dense_multiplication_helper<Coefficients>& helper,
            Fn fn, deg_t out_degree) noexcept
    {
        using key_type = tensor_basis::key_type;

        auto lhs_deg = helper.lhs_degree();
        auto rhs_deg = helper.rhs_degree();

        auto* tile = helper.write_tile();
        const auto* left_rtile = helper.left_tile();
        const auto* right_rtile = helper.right_tile();

        for (deg_t out_deg=out_degree; out_deg > 2*helper.tile_letters; --out_deg) {
            const auto stride = helper.stride(out_deg);
            const auto adj_deg = out_deg - 2*helper.tile_letters;

            // end is not actually a valid key, but it serves as a marker.
            key_type start{adj_deg, 0}, end{adj_deg, helper.range_size(adj_deg)};

            for (auto k = start; k<end; ++k) {
                auto k_reverse = helper.reverse(k);

                helper.write_tile_in(k, k_reverse);

                {
                    const auto& lhs_unit = helper.lhs_unit();
                    const auto* rhs_ptr = helper.right_fwd_read(k);

                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_unit*rhs_ptr[i*stride+j]);
                        }
                    }
                }

                {
                    const auto* lhs_ptr = helper.lhs_fwd_read(k);
                    const auto& rhs_unit = helper.rhs_unit();
                    for (dimn_t i = 0; i<helper.tile_width; ++i) {
                        for (dimn_t j = 0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_ptr[i*stride+j]*rhs_unit);
                        }
                    }
                }

                for (deg_t lh_deg=1; lh_deg < helper.tile_letters; ++lh_deg) {
                    auto rh_deg = adj_deg - lh_deg;
                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        const auto split = helper.split_key(k);
                        const auto& lhs_val = *helper.left_fwd_read(split.first);
                        helper.read_right_tile(helper.combine(key_type(helper.tile_letters-lh_deg, split.first), k));
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_val*right_rtile[j]);
                        }
                    }
                }

                for (deg_t lh_deg=0; lh_deg<adj_deg; ++lh_deg) {
                    const auto rh_deg = adj_deg - lh_deg;
                    auto split = helper.split_key(rh_deg, k);
                    helper.read_left_tile(helper.reverse(split.first));
                    helper.read_right_tile(split.second);

                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(left_rtile[i]*right_rtile[j]);
                        }
                    }
                }

                for (deg_t rh_deg=1; rh_deg<helper.tile_letters; ++rh_deg) {
                    const auto lh_deg = adj_deg - rh_deg;
                    for (dimn_t j=0; j<helper.tile_width; ++j) {
                        const auto split = helper.split_key(key_type(rh_deg, j));
                        const auto& rhs_val = *helper.right_fwd_read(helper.combine(k_reverse, helper.reverse(split.first)));
                        helper.read_left_tile(split.second);

                        for (dimn_t i=0; i<helper.tile_width; ++i) {
                            tile[i*helper.tile_width+j] += fn(left_rtile[i]*rhs_val);
                        }
                    }
                }

                helper.write_tile_out(k, k_reverse);
            }

        }

        fma_dense_traditional(helper, fn, 2*helper.tile_letters);
    }

    template <typename Coefficients, typename Alloc, typename Fn>
    void fma_dense(
            free_tensor<Coefficients, dense_vector>& result,
            const free_tensor<Coefficients, dense_vector>& lhs,
            const free_tensor<Coefficients, dense_vector>& rhs,
            Fn fn,
            deg_t max_deg
            )
    {
        using key_type = typename tensor_basis::key_type;
        if (max_deg>result.depth()) {
            max_deg = result.depth();
        }

        auto lhs_deg = lhs.degree();
        auto rhs_deg = rhs.degree();

        deg_t out_degree = std::min(max_deg, lhs_deg+rhs_deg);
        const tensor_basis* basis = &result.basis();
        result.resize(basis->size(out_degree));

        dtl::dense_multiplication_helper<Coefficients> helper(result, lhs, rhs);
        if (out_degree > 2*helper.tile_letters) {
            fma_dense_giles(helper, fn, out_degree);
        } else {
            fma_dense_traditional(helper, fn, out_degree);
        }

    }


public:

    using compatible_bases = boost::mpl::vector<tensor_basis>;

    template <typename Result, typename Vector1, typename Vector2, typename Fn>
    LAL_TENSOR_COMPAT_RVV(Result, Vector1, Vector2)
    multiply_and_add(Result& result, const Vector1& lhs, const Vector2& rhs, Fn op)
    {

       using key_type = tensor_basis::key_type;

        auto lhs_deg = helper.lhs_degree();
        auto rhs_deg = helper.rhs_degree();

        auto* tile = helper.write_tile();
        const auto* left_rtile = helper.left_tile();
        const auto* right_rtile = helper.right_tile();

        for (deg_t out_deg=out_degree; out_deg > 2*helper.tile_letters; --out_deg) {
            const auto stride = helper.stride(out_deg);
            const auto adj_deg = out_deg - 2*helper.tile_letters;

            // end is not actually a valid key, but it serves as a marker.
            key_type start{adj_deg, 0}, end{adj_deg, helper.range_size(adj_deg)};

            for (auto k = start; k<end; ++k) {
                auto k_reverse = helper.reverse(k);

                helper.write_tile_in(k, k_reverse);

                {
                    const auto& lhs_unit = helper.lhs_unit();
                    const auto* rhs_ptr = helper.right_fwd_read(k);

                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_unit*rhs_ptr[i*stride+j]);
                        }
                    }
                }

                {
                    const auto* lhs_ptr = helper.lhs_fwd_read(k);
                    const auto& rhs_unit = helper.rhs_unit();
                    for (dimn_t i = 0; i<helper.tile_width; ++i) {
                        for (dimn_t j = 0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_ptr[i*stride+j]*rhs_unit);
                        }
                    }
                }

                for (deg_t lh_deg=1; lh_deg < helper.tile_letters; ++lh_deg) {
                    auto rh_deg = adj_deg - lh_deg;
                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        const auto split = helper.split_key(k);
                        const auto& lhs_val = *helper.left_fwd_read(split.first);
                        helper.read_right_tile(helper.combine(key_type(helper.tile_letters-lh_deg, split.first), k));
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(lhs_val*right_rtile[j]);
                        }
                    }
                }

                for (deg_t lh_deg=0; lh_deg<adj_deg; ++lh_deg) {
                    const auto rh_deg = adj_deg - lh_deg;
                    auto split = helper.split_key(rh_deg, k);
                    helper.read_left_tile(helper.reverse(split.first));
                    helper.read_right_tile(split.second);

                    for (dimn_t i=0; i<helper.tile_width; ++i) {
                        for (dimn_t j=0; j<helper.tile_width; ++j) {
                            tile[i*helper.tile_width+j] += fn(left_rtile[i]*right_rtile[j]);
                        }
                    }
                }

                for (deg_t rh_deg=1; rh_deg<helper.tile_letters; ++rh_deg) {
                    const auto lh_deg = adj_deg - rh_deg;
                    for (dimn_t j=0; j<helper.tile_width; ++j) {
                        const auto split = helper.split_key(key_type(rh_deg, j));
                        const auto& rhs_val = *helper.right_fwd_read(helper.combine(k_reverse, helper.reverse(split.first)));
                        helper.read_left_tile(split.second);

                        for (dimn_t i=0; i<helper.tile_width; ++i) {
                            tile[i*helper.tile_width+j] += fn(left_rtile[i]*rhs_val);
                        }
                    }
                }

                helper.write_tile_out(k, k_reverse);
            }

        }

        fma_dense_traditional(helper, fn, 2*helper.tile_letters);}

    template <typename Coefficients, typename Alloc, typename Fn>
    void multiply_inplace(
            dense_vector_view<tensor_basis, Coefficients, Alloc>& lhs,
            const dense_vector_view<tensor_basis, Coefficients, Alloc>& rhs,
            Fn op) noexcept
    {

    }

};
#endif

class LIBALGEBRA_LITE_EXPORT left_half_shuffle_tensor_multiplier
    : public base_multiplier<left_half_shuffle_tensor_multiplier, tensor_basis>
{
    using base_type = base_multiplier<
            left_half_shuffle_tensor_multiplier, tensor_basis>;

    using typename base_type::key_type;
    using typename base_type::product_type;
    using typename base_type::reference;

    using parent_type = std::pair<key_type, key_type>;

    mutable std::unordered_map<
            parent_type, product_type, boost::hash<parent_type>>
            m_cache;
    mutable std::recursive_mutex m_lock;

    product_type
    key_prod_impl(const tensor_basis& basis, key_type lhs, key_type rhs) const;


protected:
    product_type
    shuffle(const tensor_basis& basis, key_type lhs, key_type rhs) const;

public:
    using basis_type = tensor_basis;

    explicit left_half_shuffle_tensor_multiplier(deg_t width)
    {}

    reference
    operator()(const tensor_basis& basis, key_type lhs, key_type rhs) const;
};

class LIBALGEBRA_LITE_EXPORT right_half_shuffle_tensor_multiplier
    : public base_multiplier<right_half_shuffle_tensor_multiplier, tensor_basis>
{
    using base_type = base_multiplier<
            right_half_shuffle_tensor_multiplier, tensor_basis>;

    using typename base_type::key_type;
    using typename base_type::product_type;
    using typename base_type::reference;

    using parent_type = std::pair<key_type, key_type>;

    mutable std::unordered_map<
            parent_type, product_type, boost::hash<parent_type>>
            m_cache;
    mutable std::recursive_mutex m_lock;

    product_type
    key_prod_impl(const tensor_basis& basis, key_type lhs, key_type rhs) const;

    parent_type
    split_at_right(const tensor_basis& basis, key_type key) const noexcept;


protected:
    product_type
    shuffle(const tensor_basis& basis, key_type lhs, key_type rhs) const;

public:
    using basis_type = tensor_basis;

    explicit right_half_shuffle_tensor_multiplier(deg_t width)
    {}

    reference
    operator()(const tensor_basis& basis, key_type lhs, key_type rhs) const;
};

using half_shuffle_tensor_multiplier LAL_UNUSED
        = left_half_shuffle_tensor_multiplier;

class LIBALGEBRA_LITE_EXPORT shuffle_tensor_multiplier
    : protected left_half_shuffle_tensor_multiplier
{
    using base_type = base_multiplier<
            left_half_shuffle_tensor_multiplier, tensor_basis>;
    using half_type = left_half_shuffle_tensor_multiplier;

public:
    using basis_type = tensor_basis;

    using half_type::half_type;

    typename base_type::product_type operator()(
            const tensor_basis& basis, typename base_type::key_type lhs,
            typename base_type::key_type rhs
    ) const;
};

using left_half_shuffle_multiplication
        = base_multiplication<left_half_shuffle_tensor_multiplier>;
using half_shuffle_multiplication LAL_UNUSED = left_half_shuffle_multiplication;
using right_half_shuffle_multiplication
        = base_multiplication<right_half_shuffle_tensor_multiplier>;
using shuffle_tensor_multiplication
        = base_multiplication<shuffle_tensor_multiplier>;

LAL_EXPORT_TEMPLATE_CLASS(
        multiplication_registry, left_half_shuffle_multiplication
)
LAL_EXPORT_TEMPLATE_CLASS(
        multiplication_registry, right_half_shuffle_multiplication
)
LAL_EXPORT_TEMPLATE_CLASS(
        multiplication_registry, shuffle_tensor_multiplication
)

template <
        typename Coefficients, template <typename, typename> class VectorType,
        template <typename> class StorageModel>
class shuffle_tensor
    : public unital_algebra<
              tensor_basis, Coefficients, shuffle_tensor_multiplication,
              VectorType, StorageModel>
{
    using algebra_type = unital_algebra<
            tensor_basis, Coefficients, shuffle_tensor_multiplication,
            VectorType, StorageModel>;

public:
    using algebra_type::algebra_type;

    shuffle_tensor create_alike() const {
        return shuffle_tensor(this->get_basis(), this->multiplication());
    }
};

template <typename LTensor, typename RTensor>
inline LTensor
left_half_shuffle_multiply(const LTensor& left, const RTensor& right)
{

    const auto lhsm
            = multiplication_registry<left_half_shuffle_multiplication>::get(
                    left.basis()
            );
    return multiply(*lhsm, left, right);
}

template <typename LTensor, typename RTensor>
inline LTensor
right_half_shuffle_multiply(const LTensor& left, const RTensor& right)
{
    auto rhsm = multiplication_registry<right_half_shuffle_multiplication>::get(
            left.basis()
    );
    return multiply(*rhsm, left, right);
}

template <typename LTensor, typename RTensor>
inline LTensor half_shuffle_multiply(const LTensor& left, const RTensor& right)
{
    const auto hsm = multiplication_registry<half_shuffle_multiplication>::get(
            left.basis()
            );
    return multiply(*hsm, left, right);
}

template <typename LTensor, typename RTensor>
inline LTensor shuffle_multiply(const LTensor& left, const RTensor& right)
{
    const auto sm = multiplication_registry<shuffle_tensor_multiplication>::get(
            left.basis()
    );
    return multiply(*sm, left, right);
}

}// namespace lal

#endif// LIBALGEBRA_LITE_INCLUDE_LIBALGEBRA_LITE_SHUFFLE_TENSOR_H
