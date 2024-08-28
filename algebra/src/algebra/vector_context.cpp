//
// Created by sam on 24/07/24.
//

#include "vector.h"

#include "dense_vector_factory.h"

using namespace rpy;
using namespace rpy::algebra;

using scalars::ScalarVector;

VectorData::~VectorData() = default;

Rc<VectorData> VectorData::empty_like() const noexcept
{
    return new VectorData(p_basis);
}

bool VectorData::is_sparse() const noexcept { return false; }

void VectorData::resize_by_dim(Vector& dst, dimn_t base_dim, dimn_t fibre_dim)
{
    if (dst.base_data().size() < base_dim) { dst.resize_base_dim(base_dim); }
    if (dst.fibre_data().size() < fibre_dim) {
        dst.resize_fibre_dim(fibre_dim);
    }
}

void VectorData::resize_for_operands(
        Vector& dst,
        const Vector& lhs,
        const Vector& rhs
)
{
    auto base_dim = std::max(lhs.base_data().size(), rhs.base_data().size());
    auto fibre_dim = std::max(lhs.fibre_data().size(), rhs.fibre_data().size());

    resize_by_dim(dst, base_dim, fibre_dim);
}

optional<dimn_t>
VectorData::get_index(const Vector& vector, const BasisKey& key) const noexcept
{
    optional<dimn_t> result{};
    try {
        if (const auto index = p_basis->to_index(key);
            index < dimension(vector)) {
            result = index;
        }
    } catch (...) {}

    return result;
}

Rc<VectorData> VectorData::copy() const { return new VectorData(*this); }

VectorIterator
VectorData::begin_iterator(typename scalars::ScalarVector::const_iterator it
) const
{
    return {std::move(it), p_basis->iterate_keys().begin()};
}

VectorIterator
VectorData::end_iterator(typename scalars::ScalarVector::const_iterator it
) const
{
    return {std::move(it), p_basis->iterate_keys().end()};
}

dimn_t VectorData::size(const Vector& vector) const noexcept
{
    return vector.ScalarVector::size();
}
dimn_t VectorData::dimension(const Vector& vector) const noexcept
{
    return vector.ScalarVector::dimension();
}

void VectorData::unary_inplace(
        const scalars::UnaryVectorOperation& operation,
        Vector& arg,
        const scalars::ops::Operator& op
)
{
    operation.eval_inplace(arg, op);
}

void VectorData::unary(
        const scalars::UnaryVectorOperation& operation,
        Vector& dst,
        const Vector& arg,
        const scalars::ops::Operator& op
)
{
    resize_by_dim(dst, arg.base_dimension(), arg.fibre_dimension());
    operation.eval(dst, arg, op);
}

void VectorData::binary_inplace(
        const scalars::BinaryVectorOperation& operation,
        Vector& left,
        const Vector& right,
        const scalars::ops::Operator& op
)
{
    resize_for_operands(left, left, right);
    operation.eval_inplace(left, right, op);
}

void VectorData::binary(
        const scalars::BinaryVectorOperation& operation,
        Vector& dst,
        const Vector& left,
        const Vector& right,
        const scalars::ops::Operator& op
)
{
    resize_for_operands(dst, left, right);
    operation.eval(dst, left, right, op);
}

bool VectorData::is_equal(const Vector& left, const Vector& right)
        const noexcept
{
    if (right.is_sparse()) { return right.p_context->is_equal(left, right); }
    return static_cast<const ScalarVector&>(left)
            == static_cast<const ScalarVector&>(right);
}
