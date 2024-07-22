//
// Created by sam on 7/4/24.
//

#ifndef ROUGHPY_DEVICES_KERNEL_OPERATORS_H
#define ROUGHPY_DEVICES_KERNEL_OPERATORS_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "value.h"

namespace rpy {
namespace devices {
namespace operators {

class ROUGHPY_DEVICES_EXPORT Operator
{
public:
    enum class Kind
    {
        Identity = 0,
        UnaryMinus,
        Addition,
        Subtraction,
        LeftMultiply,
        RightMultiply,
        FusedLeftMultiplyAdd,
        FusedRightMultiplyAdd,
        FusedLeftMultiplySub,
        FusedRightMultiplySub
    };

    static constexpr Kind Identity = Kind::Identity;
    static constexpr Kind UnaryMinus = Kind::UnaryMinus;
    static constexpr Kind Addition = Kind::Addition;
    static constexpr Kind Subtraction = Kind::Subtraction;
    static constexpr Kind LeftMultiply = Kind::LeftMultiply;
    static constexpr Kind RightMultiply = Kind::RightMultiply;
    static constexpr Kind FusedLeftMultiplyAdd = Kind::FusedLeftMultiplyAdd;
    static constexpr Kind FusedRightMultiplyAdd = Kind::FusedRightMultiplyAdd;
    static constexpr Kind FusedLeftMultiplySub = Kind::FusedLeftMultiplySub;
    static constexpr Kind FusedRightMultiplySub = Kind::FusedRightMultiplySub;

    virtual ~Operator();

    RPY_NO_DISCARD virtual Kind kind() const noexcept = 0;
};

class ROUGHPY_DEVICES_EXPORT IdentityOperator : public Operator
{
public:
    // template <typename T>
    // using op_type = Identity<T>;
    static constexpr Kind op_kind = Identity;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; };

    ConstReference operator()(ConstReference value) const noexcept
    {
        return value;
    }
};

class ROUGHPY_DEVICES_EXPORT UnaryMinusOperator : public Operator
{
public:
    // template <typename T>
    // using op_type = Uminus<T>;
    static constexpr Kind op_kind = UnaryMinus;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; };

    Value operator()(ConstReference value) const noexcept { return -value; }
};

class ROUGHPY_DEVICES_EXPORT AdditionOperator : public Operator
{
public:
    // template <typename T>
    // using op_type = Add<T>;
    static constexpr Kind op_kind = Addition;
    Kind kind() const noexcept override { return op_kind; }

    Value operator()(ConstReference a, ConstReference b) const noexcept
    {
        return a + b;
    }
};

class ROUGHPY_DEVICES_EXPORT SubtractionOperator : public Operator
{
public:
    static constexpr Kind op_kind = Subtraction;
    Kind kind() const noexcept override { return op_kind; }

    Value operator()(ConstReference a, ConstReference b) const noexcept
    {
        return a - b;
    }
};

class ROUGHPY_DEVICES_EXPORT LeftMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit LeftMultiplyOperator(ConstReference value) : multiplier(value) {}

    // template <typename T>
    // using op_type = LeftMultiply<T>;

    static constexpr Kind op_kind = LeftMultiply;
    Kind kind() const noexcept override { return op_kind; }
    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference value) const noexcept
    {
        return multiplier * value;
    }
};

class ROUGHPY_DEVICES_EXPORT RightMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit RightMultiplyOperator(ConstReference value) : multiplier(value) {}

    // template <typename T>
    // using op_type = RightMultiply<T>;

    static constexpr Kind op_kind = RightMultiply;
    Kind kind() const noexcept override { return op_kind; }
    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference value) const noexcept
    {
        return value * multiplier;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedLeftMultiplyOperator(ConstReference value) : multiplier(value)
    {}

    // template <typename T>
    // using op_type = FusedLeftMultiplyAdd<T>;

    static constexpr Kind op_kind = FusedLeftMultiplyAdd;
    Kind kind() const noexcept override { return op_kind; }
    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left + multiplier * right;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedRightMultiplyOperator(ConstReference value)
        : multiplier(value)
    {}

    // template <typename T>
    // using op_type = FusedRightMultiplyAdd<T>;

    static constexpr Kind op_kind = FusedRightMultiplyAdd;
    Kind kind() const noexcept override { return op_kind; }
    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left + right * multiplier;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedLeftMultiplySubOperator(ConstReference value)
        : multiplier(value)
    {}

    // template <typename T>
    // using op_type = FusedLeftMultiplySub<T>;

    static constexpr Kind op_kind = FusedLeftMultiplySub;
    Kind kind() const noexcept override { return op_kind; }

    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left - multiplier * right;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedRightMultiplySubOperator(ConstReference value)
        : multiplier(value)
    {}

    // template <typename T>
    // using op_type = FusedRightMultiplySub<T>;

    static constexpr Kind op_kind = FusedRightMultiplySub;
    Kind kind() const noexcept override { return op_kind; }
    const ConstReference& data() const noexcept  { return multiplier; }

    Value operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left - right * multiplier;
    }
};


template <typename Op>
enable_if_t<is_base_of_v<Operator, Op>, const Op&> op_cast(const Operator& op)
{
    RPY_CHECK(op.kind() == Op::op_kind);
    return static_cast<const Op&>(op);
}

}// namespace operators
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_KERNEL_OPERATORS_H
