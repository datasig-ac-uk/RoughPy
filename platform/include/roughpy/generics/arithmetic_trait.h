//
// Created by sam on 14/11/24.
//

#ifndef ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
#define ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H

#include "roughpy/core/debug_assertion.h"

#include "builtin_trait.h"
#include "type_ptr.h"

namespace rpy::generics {

class ConstRef;
class Ref;
class Value;

/**
 * @class ArithmeticTrait
 * @brief Trait providing basic arithmetic operations for value types.
 *
 * The ArithmeticTrait class defines a set of arithmetic operations that can be
 * performed on certain value types. These operations include addition,
 * subtraction, multiplication, and division. The trait is designed to be
 * inherited by specific types that need to support arithmetic operations.
 */
class ROUGHPY_PLATFORM_EXPORT ArithmeticTrait : public BuiltinTrait {
    // Builtin traits should be held on the Type class itself, so having
    // TypePtrs here would cause a reference loop.
    const Type* p_type;
    const Type* p_rational_type;
protected:

    explicit ArithmeticTrait(const Type* type, const Type* rational_type)
        : BuiltinTrait(my_id),
          p_type(type),
          p_rational_type(rational_type)
    {}

public:

    static constexpr auto my_id = BuiltinTraitID::Arithmetic;

    virtual ~ArithmeticTrait();

    /**
     * Checks if the specified arithmetic operation is supported.
     *
     * @param op The arithmetic operation to check.
     * @return True if the operation is supported, false otherwise.
     */
    RPY_NO_DISCARD virtual bool has_operation(ArithmeticOperation op) const noexcept;

    /**
     * @brief Performs in-place addition of two values.
     *
     * This pure virtual function takes pointers to two values, `lhs` and `rhs`,
     * and modifies the value pointed to by `lhs` by adding the value pointed to
     * by `rhs` to it. This operation does not throw exceptions.
     *
     * @param lhs Pointer to the destination value, which will be modified by
     * the addition.
     * @param rhs Pointer to the source value, which will be added to the
     * destination value.
     */
    virtual void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept = 0;
    /**
     * @brief Performs in-place subtraction of two values.
     *
     * This pure virtual function takes pointers to two values, `lhs` and `rhs`,
     * and modifies the value pointed to by `lhs` by subtracting the value
     * pointed to by `rhs` from it. This operation must be implemented by
     * subclasses.
     *
     * @param lhs Pointer to the destination value, which will be modified by
     * the subtraction.
     * @param rhs Pointer to the source value, which will be subtracted from the
     * destination value.
     */
    virtual void unsafe_sub_inplace(void* lhs, const void* rhs) const = 0;
    /**
     * @brief Performs in-place multiplication of two values.
     *
     * This pure virtual function takes pointers to two values, `lhs` and `rhs`,
     * and modifies the value pointed to by `lhs` by multiplying it by the value
     * pointed to by `rhs`. This operation must be implemented by subclasses.
     *
     * @param lhs Pointer to the destination value, which will be modified by
     * the multiplication.
     * @param rhs Pointer to the source value, which will be multiplied with the
     * destination value.
     */
    virtual void unsafe_mul_inplace(void* lhs, const void* rhs) const = 0;
    /**
     * @brief Performs in-place division of two values.
     *
     * This pure virtual function takes pointers to two values, `lhs` and `rhs`,
     * and modifies the value pointed to by `lhs` by dividing it by the value
     * pointed to by `rhs`. This operation must be implemented by subclasses.
     *
     * @param lhs Pointer to the destination value, which will be modified by
     * the division.
     * @param rhs Pointer to the source value, which will be used to divide the
     * destination value.
     */
    virtual void unsafe_div_inplace(void* lhs, const void* rhs) const = 0;

    RPY_NO_DISCARD
    const Type* type() const noexcept { return p_type; }

    RPY_NO_DISCARD
    const Type* rational_type() const noexcept { return p_rational_type; }

    /**
     * @brief Performs in-place addition of two values.
     *
     * This method performs an in-place addition where the value referenced by
     * `lhs` is incremented by the value referenced by `rhs`. It ensures that
     * the addition operation is supported and that both `lhs` and `rhs` are
     * valid before performing the operation. If the types of `lhs` and `rhs`
     * are identical, it directly adds the value of `rhs` to `lhs`. If the types
     * differ, it creates a temporary value to store the converted `rhs` and
     * then performs the addition.
     *
     * @param lhs Reference to the value that will be modified by the addition.
     * @param rhs Reference to the value that will be added to the value
     * referenced by `lhs`.
     */
    void add_inplace(Ref lhs, ConstRef rhs) const;
    /**
     * @brief Performs in-place subtraction of two values.
     *
     * This method performs an in-place subtraction where the value referenced
     * by `lhs` is decremented by the value referenced by `rhs`. It checks that
     * the subtraction operation is supported and that both `lhs` and `rhs` are
     * valid before performing the operation. If the types of `lhs` and `rhs`
     * are identical, it directly subtracts the value of `rhs` from `lhs`. If
     * the types differ, it creates a temporary value to store the converted
     * `rhs` and then performs the subtraction.
     *
     * @param lhs Reference to the value that will be modified by the
     * subtraction.
     * @param rhs Reference to the value that will be subtracted from the value
     * referenced by `lhs`.
     */
    void sub_inplace(Ref lhs, ConstRef rhs) const;
    /**
     * @brief Multiplies the value of the left-hand side (lhs) reference by the
     * value of the right-hand side (rhs) reference.
     *
     * This method performs an in-place multiplication of the lhs reference by
     * the rhs reference. The operation modifies the lhs reference to hold the
     * result of the multiplication.
     *
     * @param lhs A reference to the value that will be modified by the
     * multiplication operation.
     * @param rhs A constant reference to the value by which the lhs reference
     * will be multiplied.
     */
    void mul_inplace(Ref lhs, ConstRef rhs) const;
    /**
     * @brief Divides the value of lhs by the value of rhs in place.
     *
     * This method modifies the lhs argument by dividing its value by the value
     * of rhs, storing the result back in lhs. It performs validity checks on
     * rhs, including ensuring that rhs is not zero to prevent division by zero
     * errors.
     *
     * @param lhs A reference to the value which will be divided. The result
     * will be stored in this parameter.
     * @param rhs A constant reference to the value to divide lhs by. This
     * parameter must not be zero.
     */
    void div_inplace(Ref lhs, ConstRef rhs) const;

    /**
     * @brief Adds two Value instances and returns the result.
     *
     * This method checks that the addition operation is supported and that both
     * input values are valid. It then creates a new Value instance representing
     * the sum of the two input values and returns it.
     *
     * @param lhs The left-hand side Value to add.
     * @param rhs The right-hand side Value to add.
     *
     * @return The result of adding the two input Value instances.
     */
    RPY_NO_DISCARD Value add(ConstRef lhs, ConstRef rhs) const;
    /**
     * @brief Subtracts one value from another.
     *
     * This method performs subtraction of two values, `lhs` and `rhs`, and
     * returns the result. Both input values must be valid, and the arithmetic
     * operation for subtraction must be supported.
     *
     * @param lhs The left-hand side value to be subtracted from.
     * @param rhs The right-hand side value which is subtracted from the
     * left-hand side value.
     * @return The result of subtracting `rhs` from `lhs`.
     */
    RPY_NO_DISCARD Value sub(ConstRef lhs, ConstRef rhs) const;
    /**
     * @brief Multiplies two constant references to Value objects.
     *
     * This function performs a multiplication operation between two constant
     * references to Value objects and returns the result of the multiplication
     * as a new Value object.
     *
     * @param lhs The left-hand side operand of the multiplication.
     * @param rhs The right-hand side operand of the multiplication.
     * @return The result of the multiplication as a Value object.
     */
    RPY_NO_DISCARD Value mul(ConstRef lhs, ConstRef rhs) const;
    /**
     * @brief Performs division operation between two values.
     *
     * This method takes two constant references to values, performs the
     * division operation, and returns the resulting value. It first verifies
     * the availability of the division operation and the validity of both
     * operands.
     *
     * @param lhs The left-hand side value operand.
     * @param rhs The right-hand side value operand.
     * @return The result of dividing lhs by rhs.
     */
    RPY_NO_DISCARD Value div(ConstRef lhs, ConstRef rhs) const;

};


template <typename T, typename R=T>
/**
 * @brief Concrete implementation of ArithmeticTrait for specific types.
 *
 * The ArithmeticTraitImpl class provides concrete implementations for the
 * arithmetic operations defined in the ArithmeticTrait class. It supports
 * operations like addition, subtraction, multiplication, and division on
 * the value types specified during construction of the object.
 *
 * This class overrides the base class methods to perform arithmetic safely
 * in-place on the underlying data of the provided types.
 */
class ArithmeticTraitImpl : public ArithmeticTrait
{
public:

    ArithmeticTraitImpl(const Type* type, const Type* rational_type);

    RPY_NO_DISCARD bool has_operation(ArithmeticOperation op) const noexcept override;

    void unsafe_add_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_sub_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_mul_inplace(void* lhs, const void* rhs) const noexcept override;
    void unsafe_div_inplace(void* lhs, const void* rhs) const noexcept override;
};


namespace dtl {

template <typename U>
using add_result_t = decltype(std::declval<U&>() += std::declval<const U&>());

template <typename U>
using sub_result_t = decltype(std::declval<U&>() -= std::declval<const U&>());

template <typename U>
using mul_result_t = decltype(std::declval<U&>() *= std::declval<const U&>());

template <typename U, typename V>
using div_result_t = decltype(std::declval<U&>() /= std::declval<const V&>());

template <typename U, typename = void>
inline constexpr bool has_add_v = false;

template <typename U>
inline constexpr bool has_add_v<U, void_t<add_result_t<U>>> = true;

template <typename U, typename = void>
inline constexpr bool has_sub_v = false;

template <typename U>
inline constexpr bool has_sub_v<U, void_t<sub_result_t<U>>> = true;

template <typename U, typename = void>
inline constexpr bool has_mul_v = false;

template <typename U>
inline constexpr bool has_mul_v<U, void_t<mul_result_t<U>>> = true;

template <typename U, typename V, typename = void>
inline constexpr bool has_div_v = false;

template <typename U, typename V>
inline constexpr bool has_div_v<U, V, void_t<div_result_t<U, V>>> = true;

}// namespace dtl


template <typename T, typename R>
ArithmeticTraitImpl<T, R>::ArithmeticTraitImpl(
        const Type* type,
        const Type* rational_type
)
    : ArithmeticTrait(type, rational_type)
{}

template <typename T, typename R>
bool ArithmeticTraitImpl<T, R>::has_operation(ArithmeticOperation op) const noexcept
{
    switch (op) {
        case ArithmeticOperation::Add:
            return dtl::has_add_v<T>;
        case ArithmeticOperation::Sub:
            return dtl::has_sub_v<T>;
        case ArithmeticOperation::Mul:
            return dtl::has_mul_v<T>;
        case ArithmeticOperation::Div:
            return dtl::has_div_v<T, R>;
    }
    RPY_UNREACHABLE_RETURN(false);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_add_inplace(void* lhs, const void*
rhs) const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) += *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_sub_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) -= *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_mul_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) *= *static_cast<const T*>(rhs);
}

template <typename T, typename R>
void ArithmeticTraitImpl<T, R>::unsafe_div_inplace(void* lhs, const void* rhs)
        const noexcept
{
    RPY_DBG_ASSERT(dtl::has_add_v<T>);
    *static_cast<T*>(lhs) /= *static_cast<const R*>(rhs);
}




}// namespace rpy::generics

#endif //ROUGHPY_GENERICS_ARITHMETIC_TRAIT_H
