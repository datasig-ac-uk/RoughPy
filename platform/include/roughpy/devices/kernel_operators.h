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

    RPY_NO_DISCARD Kind kind() const noexcept override { return Identity; };

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

    RPY_NO_DISCARD Kind kind() const noexcept override { return UnaryMinus; };

    Value operator()(ConstReference value) const noexcept { return -value; }
};

class ROUGHPY_DEVICES_EXPORT AdditionOperator : public Operator
{
public:
    // template <typename T>
    // using op_type = Add<T>;
    Kind kind() const noexcept override { return Addition; }

    Value operator()(ConstReference a, ConstReference b) const noexcept
    {
        return a + b;
    }
};

class ROUGHPY_DEVICES_EXPORT SubtractionOperator : public Operator
{
public:
    Kind kind() const noexcept override { return Subtraction; }

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

    Kind kind() const noexcept override { return LeftMultiply; }

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

    Kind kind() const noexcept override { return RightMultiply; }

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

    Kind kind() const noexcept override { return FusedLeftMultiplyAdd; }

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

    Kind kind() const noexcept override { return FusedRightMultiplyAdd; }

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

    Kind kind() const noexcept override { return FusedLeftMultiplySub; }

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

    Kind kind() const noexcept override { return FusedRightMultiplySub; }

    Value operator()(ConstReference left, ConstReference right) const noexcept
    {
        return left - right * multiplier;
    }
};

}// namespace operators
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_KERNEL_OPERATORS_H
