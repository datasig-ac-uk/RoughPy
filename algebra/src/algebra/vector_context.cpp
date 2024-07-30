//
// Created by sam on 24/07/24.
//

#include "vector.h"

#include "dense_vector_factory.h"

using namespace rpy;
using namespace rpy::algebra;

using scalars::ScalarVector;

const VectorContext& VectorContext::get_context(const Vector& vec) noexcept
{
    return *vec.p_context;
}

VectorContext::~VectorContext() = default;

Rc<VectorContext> VectorContext::empty_like() const noexcept
{
    return new VectorContext(p_basis);
}

bool VectorContext::is_sparse() const noexcept { return false; }

void VectorContext::resize_by_dim(
        Vector& dst,
        dimn_t base_dim,
        dimn_t fibre_dim
)
{
    if (dst.base_data().size() < base_dim) { dst.resize_base_dim(base_dim); }
    if (dst.fibre_data().size() < fibre_dim) {
        dst.resize_fibre_dim(fibre_dim);
    }
}

void VectorContext::resize_for_operands(
        Vector& dst,
        const Vector& lhs,
        const Vector& rhs
)
{

    auto base_dim = std::max(lhs.base_data().size(), rhs.base_data().size());
    auto fibre_dim = std::max(lhs.fibre_data().size(), rhs.fibre_data().size());

    resize_by_dim(dst, base_dim, fibre_dim);
}

optional<dimn_t> VectorContext::get_index(
        const Vector& vector,
        const BasisKey& key
) const noexcept
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

Rc<VectorContext> VectorContext::copy() const
{
    return new VectorContext(*this);
}

VectorIterator VectorContext::make_iterator(ScalarVector::iterator it) const
{

    return {};
}

VectorIterator
VectorContext::make_const_iterator(ScalarVector::const_iterator it) const
{
    return {};
}

dimn_t VectorContext::size(const Vector& vector) const noexcept
{
    return vector.ScalarVector::size();
}
dimn_t VectorContext::dimension(const Vector& vector) const noexcept
{
    return vector.ScalarVector::dimension();
}

void VectorContext::unary_inplace(
        const scalars::UnaryVectorOperation& operation,
        Vector& arg,
        const scalars::ops::Operator& op
)
{
    operation.eval_inplace(arg, op);
}

void VectorContext::unary(
        const scalars::UnaryVectorOperation& operation,
        Vector& dst,
        const Vector& arg,
        const scalars::ops::Operator& op
)
{
    resize_by_dim(dst, arg.base_dimension(), arg.fibre_dimension());
    operation.eval(dst, arg, op);
}

void VectorContext::binary_inplace(
        const scalars::BinaryVectorOperation& operation,
        Vector& left,
        const Vector& right,
        const scalars::ops::Operator& op
)
{
    resize_for_operands(left, left, right);
    operation.eval_inplace(left, right, op);
}

void VectorContext::binary(
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

bool VectorContext::is_equal(const Vector& left, const Vector& right)
        const noexcept
{
    if (right.is_sparse()) { return right.p_context->is_equal(left, right); }
    return static_cast<const ScalarVector&>(left)
            == static_cast<const ScalarVector&>(right);
}
