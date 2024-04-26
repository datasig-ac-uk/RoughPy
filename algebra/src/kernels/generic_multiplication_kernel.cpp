//
// Created by sam on 4/24/24.
//

#include "generic_multiplication_kernel.h"
#include "multiplication_impl.h"
#include "sparse_helpers.h"

using namespace rpy;
using namespace rpy::algebra;


void algebra::dtl::GenericSquareMultiplicationKernel::eval_sss(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    KeyScalarMap mapped;
    {
        const auto keys_out = out.keys().view();
        auto key_slice = keys_out.as_slice();
        mapped = preload_map(
                p_basis,
                key_slice | views::transform(ikmap),
                out.scalars()
        );
    }

    {
        const auto left_keys = right.keys().view();
        const auto left_key_slice = left_keys.as_slice();
        const auto left_ikview = views::enumerate(left_key_slice);
        const auto right_keys = right.keys().view();
        const auto right_key_slice = right_keys.as_slice();
        const auto right_ikview = views::enumerate(right_key_slice);

        if (p_basis->is_graded()) {
            triangular_ordered_generic_multiplication_right_sparse(
                    mapped,
                    m_key_func,
                    m_func,
                    p_basis,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view(),
                    p_basis->max_degree()
            );
        } else {
            square_generic_multiplication(
                    mapped,
                    m_key_func,
                    m_func,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view()
            );
        }
    }
    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_ssd(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    KeyScalarMap mapped;
    {
        const auto keys_out = out.keys().view();
        auto key_slice = keys_out.as_slice();
        mapped = preload_map(
                p_basis,
                key_slice | views::transform(ikmap),
                out.scalars()
        );
    }

    {
        const auto left_keys = right.keys().view();
        const auto left_key_slice = left_keys.as_slice();
        const auto left_ikview = views::enumerate(left_key_slice);
        const auto right_ikview
                = views::ints(static_cast<dimn_t>(0), right.size())
                | views::transform(ikmap);

        if (p_basis->is_graded()) {
            triangular_ordered_generic_multiplication_right_dense(
                    mapped,
                    m_key_func,
                    m_func,
                    p_basis,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view(),
                    p_basis->max_degree()
            );
        } else {
            square_generic_multiplication(
                    mapped,
                    m_key_func,
                    m_func,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view()
            );
        }
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_sds(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    KeyScalarMap mapped;
    {
        const auto keys_out = out.keys().view();
        auto key_slice = keys_out.as_slice();
        mapped = preload_map(
                p_basis,
                key_slice | views::transform(ikmap),
                out.scalars()
        );
    }

    {
        const auto left_ikview
                = views::ints(static_cast<dimn_t>(0), left.size())
                | views::transform(ikmap);
        const auto right_keys = right.keys().view();
        const auto right_key_slice = right_keys.as_slice();
        const auto right_ikview = views::enumerate(right_key_slice);

        if (p_basis->is_graded()) {
            triangular_ordered_generic_multiplication_right_dense(
                    mapped,
                    m_key_func,
                    m_func,
                    p_basis,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view(),
                    p_basis->max_degree()
            );
        } else {
            square_generic_multiplication(
                    mapped,
                    m_key_func,
                    m_func,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view()
            );
        }
    }
    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_sdd(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    KeyScalarMap mapped;
    {
        const auto keys_out = out.keys().view();
        auto key_slice = keys_out.as_slice();
        mapped = preload_map(
                p_basis,
                key_slice | views::transform(ikmap),
                out.scalars()
        );
    }

    {
        const auto left_ikview
                = views::ints(static_cast<dimn_t>(0), left.size())
                | views::transform(ikmap);
        const auto right_keys = right.keys().view();
        const auto right_ikview
                = views::ints(static_cast<dimn_t>(0), right.size())
                | views::transform(ikmap);

        if (p_basis->is_graded()) {
            triangular_ordered_generic_multiplication_right_dense(
                    mapped,
                    m_key_func,
                    m_func,
                    p_basis,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view(),
                    p_basis->max_degree()
            );
        } else {
            square_generic_multiplication(
                    mapped,
                    m_key_func,
                    m_func,
                    left_ikview,
                    left.scalars().view(),
                    right_ikview,
                    right.scalars().view()
            );
        }
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_dss(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    const auto left_keys = right.keys().view();
    const auto left_key_slice = left_keys.as_slice();
    const auto left_ikview = views::enumerate(left_key_slice);
    const auto right_keys = right.keys().view();
    const auto right_key_slice = right_keys.as_slice();
    const auto right_ikview = views::enumerate(right_key_slice);

    DenseVectorMap mapped(out.mut_scalars(), p_basis);
    if (p_basis->is_graded()) {
        triangular_ordered_generic_multiplication_right_sparse(
                mapped,
                m_key_func,
                m_func,
                p_basis,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                p_basis->max_degree()
        );
    } else {
        square_generic_multiplication(
                mapped,
                m_key_func,
                m_func,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view()
        );
    }
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_dsd(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    const auto left_keys = right.keys().view();
    const auto left_key_slice = left_keys.as_slice();
    const auto left_ikview = views::enumerate(left_key_slice);
    const auto right_ikview = views::ints(static_cast<dimn_t>(0), right.size())
            | views::transform(ikmap);

    DenseVectorMap mapped(out.mut_scalars(), p_basis);
    if (p_basis->is_graded()) {
        triangular_ordered_generic_multiplication_right_dense(
                mapped,
                m_key_func,
                m_func,
                p_basis,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                p_basis->max_degree()
        );
    } else {
        square_generic_multiplication(
                mapped,
                m_key_func,
                m_func,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view()
        );
    }
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_dds(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    const auto left_ikview = views::ints(static_cast<dimn_t>(0), left.size())
            | views::transform(ikmap);
    const auto right_keys = right.keys().view();
    const auto right_key_slice = right_keys.as_slice();
    auto right_ikview = views::enumerate(right_key_slice);

    DenseVectorMap mapped(out.mut_scalars(), p_basis);
    if (p_basis->is_graded()) {
        triangular_ordered_generic_multiplication_right_dense(
                mapped,
                m_key_func,
                m_func,
                p_basis,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                p_basis->max_degree()
        );
    } else {
        square_generic_multiplication(
                mapped,
                m_key_func,
                m_func,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view()
        );
    }
}
void algebra::dtl::GenericSquareMultiplicationKernel::eval_ddd(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    const auto left_ikview = views::ints(static_cast<dimn_t>(0), left.size())
            | views::transform(ikmap);
    const auto right_ikview = views::ints(static_cast<dimn_t>(0), right.size())
            | views::transform(ikmap);

    DenseVectorMap mapped(out.mut_scalars(), p_basis);
    if (p_basis->is_graded()) {
        triangular_ordered_generic_multiplication_right_dense(
                mapped,
                m_key_func,
                m_func,
                p_basis,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                p_basis->max_degree()
        );
    } else {
        square_generic_multiplication(
                mapped,
                m_key_func,
                m_func,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view()
        );
    }
}

void algebra::dtl::GenericSquareMultiplicationKernel::operator()(
        VectorData& out,
        const VectorData& left,
        const VectorData& right
) const
{
    switch (get_sparse_dense_config(out, left, right)) {
        case 0b000: eval_sss(out, left, right); break;
        case 0b001: eval_ssd(out, left, right); break;
        case 0b010: eval_sds(out, left, right); break;
        case 0b011: eval_sdd(out, left, right); break;
        case 0b100: eval_dss(out, left, right); break;
        case 0b101: eval_dsd(out, left, right); break;
        case 0b110: eval_dds(out, left, right); break;
        case 0b111: eval_ddd(out, left, right); break;
        default: break;
    }
}
