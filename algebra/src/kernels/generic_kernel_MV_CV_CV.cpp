
#include "generic_kernel_MV_CV_CV.h"

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
{}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_ssd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_sds(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_sdd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dss(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}
void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dsd(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}

void algebra::dtl::GenericKernel<
        algebra::dtl::MutableVectorArg,
        algebra::dtl::ConstVectorArg,
        algebra::dtl::ConstVectorArg>::
        eval_dds(
                VectorData& out,
                const VectorData& left,
                const VectorData& right
        ) const
{}
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
operator()(VectorData& out, VectorData& left, VectorData& right) const
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
