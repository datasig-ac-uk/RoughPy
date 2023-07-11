#ifndef ROUGHPY_DEVICE_MULTIARRAY_H_
#define ROUGHPY_DEVICE_MULTIARRAY_H_

#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/algebra/context.h>
#include <roughpy/algebra/free_tensor.h>
#include <roughpy/algebra/lie.h>
#include <roughpy/algebra/shuffle_tensor.h>

namespace rpy {
namespace device {

class RPY_EXPORT DeviceContext
{
    algebra::context_pointer p_base;
    scalars::ScalarDeviceInfo m_device;

protected:
    explicit DeviceContext(
            algebra::context_pointer&& ctx, scalars::ScalarDeviceInfo dev_info
    )
        : p_base(std::move(ctx)), m_device(std::move(dev_info))
    {}

public:
    virtual ~DeviceContext();

    RPY_NO_DISCARD
    const scalars::ScalarType* ctype() const noexcept
    {
        return p_base->ctype();
    }

    RPY_NO_DISCARD
    algebra::context_pointer context() const noexcept { return p_base; }

    virtual dimn_t count_nonzero(const scalars::ScalarArray& data) const = 0;
    virtual bool
    all_equal(const scalars::ScalarArray& data, scalars::Scalar value) const
            = 0;
    virtual bool is_zero(const scalars::ScalarArray& data) const = 0;
    virtual bool
    equals(const scalars::ScalarArray& lhs, const scalars::ScalarArray& rhs
    ) const = 0;

    enum UnaryOperation
    {
        Clone,
        UMinus,
        SMul,
        SDiv
    };

    virtual void unary_op(
            scalars::ScalarArray& result, const scalars::ScalarArray& arg,
            UnaryOperation op, optional<scalars::Scalar> scalar
    ) const = 0;

    enum InplaceUnaryOperation
    {
        InplaceUMinus,
        InplaceSMul,
        InplaceSDiv
    };

    virtual void inplace_unary_op(
            scalars::ScalarArray& arg, InplaceUnaryOperation op,
            optional<scalars::Scalar> scalar
    ) const = 0;

    enum BinaryOperation
    {
        Add,
        Sub
    };

    virtual void binary_op(
            scalars::ScalarArray& result, const scalars::ScalarArray& lhs,
            const scalars::ScalarArray& rhs, BinaryOperation op,
            optional<scalars::Scalar> scalar
    ) const = 0;

    enum InplaceBinaryOperation
    {
        InplaceAdd,
        InplaceSub,
        InplaceAddSMul,
        InplaceAddSDiv,
        InplaceSubSMul,
        InplaceSubSDiv
    };

    virtual void inplace_binary_op(
            scalars::ScalarArray& lhs, const scalars::ScalarArray& rhs,
            InplaceBinaryOperation op, optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    enum MultiplyOperation
    {
        PassThrough,
        LeftMultiply,
        RightMultiply,
        PostDivide
    };

    virtual void free_tensor_multiply(
            scalars::ScalarArray& result, const scalars::ScalarArray& lhs,
            const scalars::ScalarArray& rhs, MultiplyOperation op,
            optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    virtual void inplace_free_tensor_multiply(
            scalars::ScalarArray& lhs, const scalars::ScalarArray& rhs,
            MultiplyOperation op, optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    virtual void shuffle_tensor_multiply(
            scalars::ScalarArray& result, const scalars::ScalarArray& lhs,
            const scalars::ScalarArray& rhs, MultiplyOperation op,
            optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    virtual void inplace_shuffle_tensor_multiply(
            scalars::ScalarArray& lhs, const scalars::ScalarArray& rhs,
            MultiplyOperation op, optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    virtual void lie_multiply(
            scalars::ScalarArray& result, const scalars::ScalarArray& lhs,
            const scalars::ScalarArray& rhs, MultiplyOperation op,
            optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;

    virtual void inplace_lie_multiply(
            scalars::ScalarArray& lhs, const scalars::ScalarArray& rhs,
            MultiplyOperation op, optional<scalars::Scalar> scalar_a,
            optional<scalars::Scalar> scalar_b
    ) const = 0;
};

}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_MULTIARRAY_H_
