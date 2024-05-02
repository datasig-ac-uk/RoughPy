
#include "generic_kernel_MV_CV_CV.h"

#include "sparse_helpers.h"

using namespace rpy;
using namespace rpy::algebra;

void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_sss(
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

        binary_operation_into_map(
                mapped,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                m_func
        );
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_ssd(
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
        const scalars::Scalar zero(out.scalar_type());
        const auto left_keys = left.keys().view();
        const auto left_key_slice = left_keys.as_slice();
        const auto left_ikview = views::enumerate(left_key_slice);
        const auto right_ikview = views::ints(dimn_t(0), right.size())
                | views::transform(ikmap);

        // if the sparse (lsft) vector is applied first, then the dense
        // application will cause lots of bad insertions. So better to do the
        // dense one first.
        binary_operation_into_map_right(
                mapped,
                right_ikview,
                right.scalars().view(),
                m_func,
                zero
        );
        binary_operation_into_map_left(
                mapped,
                left_ikview,
                left.scalars().view(),
                m_func,
                zero
        );
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_sds(
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
                = views::ints(dimn_t(0), left.size()) | views::transform(ikmap);
        const auto right_keys = right.keys().view();
        const auto right_key_slice = right_keys.as_slice();
        const auto right_ikview = views::enumerate(right_key_slice);

        // The dense (left) vector goes first, which is important to avoid lots
        // of horrible inserting in the second application.
        binary_operation_into_map(
                mapped,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                m_func
        );
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_sdd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{

    auto ikmap = [basis = p_basis](dimn_t idx) {
        return std::make_tuple(idx, basis->to_key(idx));
    };

    // This probably should be dense
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
                = views::ints(dimn_t(0), left.size()) | views::transform(ikmap);
        const auto right_ikview = views::ints(dimn_t(0), right.size())
                | views::transform(ikmap);
        binary_operation_into_map(
                mapped,
                left_ikview,
                left.scalars().view(),
                right_ikview,
                right.scalars().view(),
                m_func
        );
    }

    write_sparse_result(out, mapped);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dss(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{
    auto left_dense = left.make_dense(p_basis);
    auto right_dense = right.make_dense(p_basis);
    eval_ddd(out, *left_dense, *right_dense);
}

void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dsd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{
    auto left_dense = left.make_dense(p_basis);
    eval_ddd(out, *left_dense, right);
}

void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dds(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{
    auto right_dense = right.make_dense(p_basis);
    eval_ddd(out, left, *right_dense);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_ddd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{
    if (out.empty()) {
        const auto max_size = std::max(left.size(), right.size());
        out.resize(max_size);
    }

    const auto size = std::min(left.size(), right.size());

    auto scalars_out = out.mut_scalars().mut_view();
    const auto scalars_left = left.scalars().view();
    const auto scalars_right = right.scalars().view();

    for (dimn_t i = 0; i < size; ++i) {
        auto tmp = scalars_out[i];
        m_func(tmp, scalars_left[i], scalars_right[i]);
    }

    scalars::Scalar zero(scalars_out.type());

    for (dimn_t i = size; i < left.size(); ++i) {
        auto tmp = scalars_out[i];
        m_func(tmp, scalars_left[i], zero);
    }

    for (dimn_t i = size; i < right.size(); ++i) {
        auto tmp = scalars_out[i];
        m_func(tmp, zero, scalars_right[i]);
    }
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
operator()(VectorData& out, const VectorData& left, const VectorData& right) const
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
