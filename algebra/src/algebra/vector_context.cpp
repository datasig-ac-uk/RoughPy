//
// Created by sam on 24/07/24.
//

#include "vector.h"

#include "dense_vector_factory.h"

using namespace rpy;
using namespace rpy::algebra;


namespace rpy {
namespace algebra {
class VectorIterator {};
}
}


using scalars::ScalarVector;

VectorContext::~VectorContext() = default;


Rc<VectorContext> VectorContext::empty_like() const noexcept
{
    return new VectorContext(p_basis);
}


bool VectorContext::is_sparse() const noexcept { return false; }

Rc<VectorContext> VectorContext::copy() const
{
    return new VectorContext(p_basis);
}

VectorIterator VectorContext::make_iterator(ScalarVector::iterator it
) const
{

    return {};
}

VectorIterator
VectorContext::make_const_iterator(ScalarVector::const_iterator it
) const
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
) const
{
    operation.eval(dst, arg, op);
}

void VectorContext::binary_inplace(
        const scalars::BinaryVectorOperation& operation,
        Vector& left,
        const Vector& right,
        const scalars::ops::Operator& op
)
{
    operation.eval_inplace(left, right, op);
}

void VectorContext::binary(
        const scalars::BinaryVectorOperation& operation,
        Vector& dst,
        const Vector& left,
        const Vector& right,
        const scalars::ops::Operator& op
) const
{
    operation.eval(dst, left, right, op);
}

bool VectorContext::is_equal(const Vector& left, const Vector& right)
        const noexcept
{
    if (right.is_sparse()) {
        return right.p_context->is_equal(left, right);
    }
    return static_cast<const ScalarVector&>(left) == static_cast<const ScalarVector&>(right);
}
