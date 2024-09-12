//
// Created by sam on 7/4/24.
//

#ifndef ROUGHPY_DEVICES_KERNEL_OPERATORS_H
#define ROUGHPY_DEVICES_KERNEL_OPERATORS_H

#include <roughpy/core/traits.h>
#include <roughpy/core/types.h>

#include "value.h"

#include "macros.h"

namespace rpy {
namespace devices {
namespace operators {

class ROUGHPY_DEVICES_EXPORT Operator
{
public:
    enum class Kind
    {
        Identity = 0,
        Minus,
        Addition,
        Subtraction,
        LeftMultiply,
        RightMultiply,
        FusedLeftMultiplyAdd,
        FusedRightMultiplyAdd,
        FusedLeftMultiplySub,
        FusedRightMultiplySub
    };

    static constexpr auto Identity = Kind::Identity;
    static constexpr auto Minus = Kind::Minus;
    static constexpr auto Addition = Kind::Addition;
    static constexpr auto Subtraction = Kind::Subtraction;
    static constexpr auto LeftMultiply = Kind::LeftMultiply;
    static constexpr auto RightMultiply = Kind::RightMultiply;
    static constexpr auto FusedLeftMultiplyAdd = Kind::FusedLeftMultiplyAdd;
    static constexpr auto FusedRightMultiplyAdd = Kind::FusedRightMultiplyAdd;
    static constexpr auto FusedLeftMultiplySub = Kind::FusedLeftMultiplySub;
    static constexpr auto FusedRightMultiplySub = Kind::FusedRightMultiplySub;

    virtual ~Operator();

    RPY_NO_DISCARD virtual Kind kind() const noexcept = 0;
};

class IdentityOperator;
class MinusOperator;
class LeftMultiplyOperator;
class RightMultiplyOperator;
class AdditionOperator;
class SubtractionOperator;
class FusedLeftMultiplyAddOperator;
class FusedRightMultiplyAddOperator;
class FusedLeftMultiplySubOperator;
class FusedRightMultiplySubOperator;

template <typename T>
struct ArgumentTraits {
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
};

template <>
struct ArgumentTraits<Value> {
    using value_type = Value;
    using reference = Reference;
    using const_reference = ConstReference;
};

namespace dtl {

template <Operator::Kind Kind>
struct OpKindTag {
    static constexpr Operator::Kind op_kind = Kind;
};

}// namespace dtl

template <typename T>
struct Identity : dtl::OpKindTag<Operator::Identity> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "identity";

    using abstract_type = IdentityOperator;

    RPY_NO_DISCARD RPY_HOST_DEVICE value_type operator()(const_reference arg
    ) const
    {
        return arg;
    }
};

template <typename T>
struct Minus : dtl::OpKindTag<Operator::Minus> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    static constexpr string_view name = "minus";

    using abstract_type = MinusOperator;

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(-arg))
    {
        return -arg;
    }
};

template <typename T>
struct Add : dtl::OpKindTag<Operator::Addition> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    using abstract_type = AdditionOperator;
    static constexpr string_view name = "addition";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + right))
    {
        return left + right;
    }
};

template <typename T>
struct Sub : dtl::OpKindTag<Operator::Subtraction> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    using abstract_type = SubtractionOperator;
    static constexpr string_view name = "subtraction";

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - right))
    {
        return left - right;
    }
};

template <typename T>
struct LeftScalarMultiply : dtl::OpKindTag<Operator::LeftMultiply> {

    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    using abstract_type = LeftMultiplyOperator;

    static constexpr string_view name = "left_scalar_multiply";
    constexpr explicit LeftScalarMultiply(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(data * arg))
    {
        return data * arg;
    }

private:
    const_reference data;
};

template <typename T>
struct RightScalarMultiply : dtl::OpKindTag<Operator::RightMultiply> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;

    using abstract_type = RightMultiplyOperator;

    static constexpr string_view name = "right_scalar_multiply";
    constexpr explicit RightScalarMultiply(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type
    operator()(const_reference arg) noexcept(noexcept(arg * data))
    {
        return arg * data;
    }

private:
    const_reference data;
};

template <typename T>
struct FusedLeftScalarMultiplyAdd
    : dtl::OpKindTag<Operator::FusedLeftMultiplyAdd> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    using abstract_type = FusedLeftMultiplyAddOperator;

    static constexpr string_view name = "fused_add_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplyAdd(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + data * right))
    {
        return left + data * right;
    }

private:
    const_reference data;
};

template <typename T>
struct FusedRightScalarMultiplyAdd
    : dtl::OpKindTag<Operator::FusedRightMultiplyAdd> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    using abstract_type = FusedRightMultiplyAddOperator;

    static constexpr string_view name = "fused_add_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplyAdd(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left + right * data))
    {
        return left + right * data;
    }

private:
    const_reference data;
};

template <typename T>
struct FusedLeftScalarMultiplySub
    : dtl::OpKindTag<Operator::FusedLeftMultiplySub> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    using abstract_type = FusedLeftMultiplySubOperator;

    static constexpr string_view name = "fused_sub_scalar_left_mul";
    constexpr explicit FusedLeftScalarMultiplySub(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - data * right))
    {
        return left - data * right;
    }

private:
    const_reference data;
};

template <typename T>
struct FusedRightScalarMultiplySub
    : dtl::OpKindTag<Operator::FusedRightMultiplySub> {
    using value_type = typename ArgumentTraits<T>::value_type;
    using reference = typename ArgumentTraits<T>::reference;
    using const_reference = typename ArgumentTraits<T>::const_reference;
    using abstract_type = FusedRightMultiplySubOperator;

    static constexpr string_view name = "fused_sub_scalar_right_mul";
    constexpr explicit FusedRightScalarMultiplySub(const_reference multiplier)
        : data(std::move(multiplier))
    {}

    RPY_NO_DISCARD RPY_HOST_DEVICE constexpr value_type operator()(
            const_reference left,
            const_reference right
    ) noexcept(noexcept(left - right * data))
    {
        return left - right * data;
    }

private:
    const_reference data;
};

class ROUGHPY_DEVICES_EXPORT IdentityOperator : public Operator
{
public:
    template <typename T>
    using op_type = operators::Identity<T>;

    static constexpr Kind op_kind = Identity;

    template <typename T = Value>
    op_type<T> into() const
    {
        ignore_unused(this);
        return {};
    }

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; };

    ConstReference operator()(ConstReference value) const noexcept
    {
        return value;
    }
};

class ROUGHPY_DEVICES_EXPORT MinusOperator : public Operator
{
public:
    template <typename T>
    using op_type = operators::Minus<T>;

    static constexpr Kind op_kind = Minus;

    template <typename T = Value>
    op_type<T> into() const
    {
        ignore_unused(this);
        return {};
    }

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; };

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference value) const { return -value; }
};

class ROUGHPY_DEVICES_EXPORT AdditionOperator : public Operator
{
public:
    template <typename T>
    using op_type = operators::Add<T>;

    static constexpr Kind op_kind = Addition;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }

    template <typename T = Value>
    op_type<T> into() const
    {
        ignore_unused(this);
        return {};
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs + rhs;
    }
};

class ROUGHPY_DEVICES_EXPORT SubtractionOperator : public Operator
{
public:
    template <typename T>
    using op_type = operators::Sub<T>;

    static constexpr Kind op_kind = Subtraction;
    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }

    template <typename T = Value>
    op_type<T> into() const
    {
        ignore_unused(this);
        return {};
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs - rhs;
    }
};

class ROUGHPY_DEVICES_EXPORT LeftMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit LeftMultiplyOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::LeftScalarMultiply<T>;

    static constexpr Kind op_kind = LeftMultiply;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }

    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference value) const { return multiplier * value; }
};

class ROUGHPY_DEVICES_EXPORT RightMultiplyOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit RightMultiplyOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::RightScalarMultiply<T>;

    static constexpr Kind op_kind = RightMultiply;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }
    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference value) const { return value * multiplier; }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplyAddOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedLeftMultiplyAddOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::FusedLeftScalarMultiplyAdd<T>;

    static constexpr Kind op_kind = FusedLeftMultiplyAdd;

    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }
    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs + multiplier * rhs;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplyAddOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedRightMultiplyAddOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::FusedRightScalarMultiplyAdd<T>;

    static constexpr Kind op_kind = FusedRightMultiplyAdd;
    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }
    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs + rhs * multiplier;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedLeftMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedLeftMultiplySubOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::FusedLeftScalarMultiplySub<T>;

    static constexpr Kind op_kind = FusedLeftMultiplySub;
    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }

    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs - multiplier * rhs;
    }
};

class ROUGHPY_DEVICES_EXPORT FusedRightMultiplySubOperator : public Operator
{
    ConstReference multiplier;

public:
    explicit FusedRightMultiplySubOperator(ConstReference value)
        : multiplier(std::move(value))
    {}

    template <typename T>
    using op_type = operators::FusedRightScalarMultiplySub<T>;

    static constexpr Kind op_kind = FusedRightMultiplySub;
    RPY_NO_DISCARD Kind kind() const noexcept override { return op_kind; }
    RPY_NO_DISCARD const ConstReference& data() const noexcept
    {
        return multiplier;
    }

    template <typename T = Value>
    op_type<T> into() const
    {
        if constexpr (is_same_v<T, Value>) {
            return op_type<T>(multiplier);
        } else {
            return op_type<T>(value_cast<T>(multiplier));
        }
    }

    // ReSharper disable once CppPassValueParameterByConstReference
    Value operator()(ConstReference lhs, ConstReference rhs) const
    {
        return lhs - rhs * multiplier;
    }
};

template <typename Op>
// NOLINTNEXTLINE(*-identifier-length)
enable_if_t<is_base_of_v<Operator, Op>, const Op&> op_cast(const Operator& op)
{
    RPY_CHECK(op.kind() == Op::op_kind);
    return static_cast<const Op&>(op);
}

}// namespace operators
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICES_KERNEL_OPERATORS_H
