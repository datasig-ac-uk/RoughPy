#include "generic_kernel_MV_CV_CS.h"

#include "sparse_helpers.h"

using namespace rpy;
using namespace rpy::algebra;

void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstScalarArg>::
        eval_sparse_sparse(
                VectorData& out,
                const VectorData& arg,
                const scalars::Scalar& scal
        ) const
{
    const auto scalars_in = arg.scalars().view();
    const auto keys_in = arg.keys().view();

    KeyScalarMap tmp_map;
    {
        const auto keys_out = out.keys().view();
        const auto scalars_out = out.scalars().view();

        auto out_key_view = keys_out.as_slice();
        preload_map(tmp_map, views::enumerate(out_key_view), scalars_out);

        auto in_key_view = keys_in.as_slice();
        write_with_sparse(
                tmp_map,
                views::enumerate(in_key_view),
                scalars_in,
                [&](scalars::Scalar& left, const scalars::Scalar& right) {
                    m_func(left, right, scal);
                }
        );
    }

    write_sparse_result(out, tmp_map);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstScalarArg>::
        eval_sparse_dense(
                VectorData& out,
                const VectorData& arg,
                const scalars::Scalar& scal
        ) const
{
    const auto size = arg.size();

    const auto scalars_in = arg.scalars().view();

    KeyScalarMap tmp_map;
    {
        // We need to make sure the scalar views are unmapped before the writing
        // happens later
        const auto out_keys = out.keys().view();
        const auto out_scalars = out.scalars().view();

        auto key_view = out_keys.as_slice();
        preload_map(tmp_map, views::enumerate(key_view), out_scalars);

        write_with_sparse(
                tmp_map,
                views::ints(dimn_t(0), size)
                        | views::transform([basis = p_basis](dimn_t idx) {
                              return std::make_tuple(idx, basis->to_key(idx));
                          }),
                scalars_in,
                [&](scalars::Scalar& left, const scalars::Scalar& right) {
                    m_func(left, right, scal);
                }
        );
    }

    write_sparse_result(out, tmp_map);
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstScalarArg>::
        eval_dense_sparse(
                VectorData& out,
                const VectorData& arg,
                const scalars::Scalar& scal
        ) const
{
    RPY_CHECK(p_basis->is_ordered());

    const auto size = arg.size();
    const auto keys_in = arg.keys().view();
    const auto scalars_in = arg.scalars().view();

    RPY_DBG_ASSERT(size > 0);
    const auto max_size
            = p_basis->dense_dimension(p_basis->to_index(keys_in[size - 1]));

    out.resize(max_size);

    auto scalars_out = out.mut_scalars().mut_view();
    for (dimn_t i = 0; i < size; ++i) {
        auto tmp = scalars_out[p_basis->to_index(keys_in[i])];
        m_func(tmp, scalars_in[i], scal);
    }
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstScalarArg>::
        eval_dense_dense(
                VectorData& out,
                const VectorData& arg,
                const scalars::Scalar& scal
        ) const
{

    const auto size = std::min(out.size(), arg.size());
    auto scalars_out = out.mut_scalars().mut_view();
    const auto scalars_in = arg.scalars().view();

    for (dimn_t i = 0; i < size; ++i) {
        auto tmp = scalars_out[i];
        m_func(tmp, scalars_in[i], scal);
    }
}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstScalarArg>::
operator()(VectorData& out, const VectorData& arg, const scalars::Scalar& scal)
        const
{
    switch (get_sparse_dense_config(out, arg)) {
        case 0b00: eval_sparse_sparse(out, arg, scal); break;
        case 0b01: eval_sparse_dense(out, arg, scal); break;
        case 0b10: eval_dense_sparse(out, arg, scal); break;
        case 0b11: eval_dense_dense(out, arg, scal); break;
        default: break;
    }
}
