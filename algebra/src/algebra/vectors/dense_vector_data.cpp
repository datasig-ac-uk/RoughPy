//
// Created by sam on 8/28/24.
//

#include "vector.h"

using namespace rpy;
using namespace rpy::algebra;

Rc<VectorData> DenseVectorData::empty_like() const noexcept
{
    return new DenseVectorData(basis());
}
bool DenseVectorData::is_sparse() const noexcept
{
    return VectorData::is_sparse();
}
bool DenseVectorData::is_contiguous() const noexcept
{
    return VectorData::is_contiguous();
}
void DenseVectorData::resize_by_dim(dimn_t base_dim, dimn_t fibre_dim) {}
void DenseVectorData::resize_for_operands(
        const VectorData& lhs,
        const VectorData& rhs
)
{}
optional<dimn_t> DenseVectorData::get_index(BasisKeyCRef key) const noexcept {}

scalars::ScalarVector& DenseVectorData::as_mut_scalar_vector()
{
    return m_vector;
}
const scalars::ScalarVector& DenseVectorData::as_scalar_vector() const
{
    return m_vector;
}
Rc<VectorData> DenseVectorData::copy() const
{
    return new DenseVectorData(*this);
}
VectorIterator DenseVectorData::begin() const
{
    return {m_vector.begin(), basis()->keys_begin()};
}
VectorIterator DenseVectorData::end() const
{
    return {m_vector.end(), basis()->keys_end()};
}
dimn_t DenseVectorData::size() const noexcept { return m_vector.size(); }
dimn_t DenseVectorData::dimension() const noexcept
{
    return m_vector.dimension();
}
void DenseVectorData::unary_inplace(
        const scalars::UnaryVectorOperation& operation,
        const scalars::ops::Operator& op
)
{
    operation.eval_inplace(m_vector, op);
}
void DenseVectorData::unary(
        const scalars::UnaryVectorOperation& operation,
        const VectorData& arg,
        const scalars::ops::Operator& op
)
{
    operation.eval(m_vector, arg.as_scalar_vector(), op);
}
void DenseVectorData::binary_inplace(
        const scalars::BinaryVectorOperation& operation,
        const VectorData& right,
        const scalars::ops::Operator& op
)
{
    operation.eval_inplace(m_vector, right.as_scalar_vector(), op);
}
void DenseVectorData::binary(
        const scalars::BinaryVectorOperation& operation,
        const VectorData& left,
        const VectorData& right,
        const scalars::ops::Operator& op
)
{
    operation.eval(
            m_vector,
            left.as_scalar_vector(),
            right.as_scalar_vector(),
            op
    );
}
bool DenseVectorData::is_equal(const VectorData& right) const noexcept
{
    if (basis() == right.basis() && !right.is_sparse()
        && right.is_contiguous()) {
        return m_vector == right.as_scalar_vector();
    }
    return false;
}
