//
// Created by sammorley on 16/11/24.
//

#include "roughpy/generics/values.h"

#include "roughpy/generics/arithmetic_trait.h"
#include "roughpy/generics/builtin_trait.h"
#include "roughpy/generics/type.h"

#include <boost/mpl/integral_c_tag.hpp>
#include <boost/url/detail/any_params_iter.hpp>

using namespace rpy;
using namespace rpy::generics;

namespace {

const ArithmeticTrait* arithmetic_trait(const Type* tp)
{
    return trait_cast<ArithmeticTrait>(
            tp->get_builtin_trait(BuiltinTraitID::Arithmetic)
    );
}

template <typename F>
void value_arithmetic_impl(
        F&& fn,
        const ArithmeticTrait* trait,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    if (RPY_LIKELY(ltype == rtype)) {
        fn(trait, lvalue, rvalue);
        return;
    }

    Value tmp(ltype);
    tmp = ConstRef(rtype, rvalue);

    fn(trait, lvalue, tmp.data());
}

void value_arithmetic_add(
        const ArithmeticTrait* trait,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    value_arithmetic_impl(
            [](const ArithmeticTrait* trait, void* lptr, const void* rptr) {
                trait->unsafe_add_inplace(lptr, rptr);
            },
            trait,
            ltype,
            lvalue,
            rtype,
            rvalue
    );
}

void value_arithmetic_subtract(
        const ArithmeticTrait* trait,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    value_arithmetic_impl(
            [](const ArithmeticTrait* trait, void* lptr, const void* rptr) {
                trait->unsafe_sub_inplace(lptr, rptr);
            },
            trait,
            ltype,
            lvalue,
            rtype,
            rvalue
    );
}

void value_arithmetic_multiply(
        const ArithmeticTrait* trait,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    value_arithmetic_impl(
            [](const ArithmeticTrait* trait, void* lptr, const void* rptr) {
                trait->unsafe_mul_inplace(lptr, rptr);
            },
            trait,
            ltype,
            lvalue,
            rtype,
            rvalue
    );
}

void value_arithmetic_divide(
        const ArithmeticTrait* trait,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    if (rtype == trait->rational_type()) {
        trait->unsafe_div_inplace(lvalue, rvalue);
        return;
    }

    Value tmp(trait->rational_type());
    tmp = ConstRef(rtype, rvalue);

    trait->unsafe_div_inplace(tmp.data(), tmp.data());
}



}// namespace

void generics::dtl::value_inplace_arithmetic(
        ArithmeticOperation operation,
        const Type* ltype,
        void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    auto trait = arithmetic_trait(ltype);
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(operation));

    switch (operation) {
        case ArithmeticOperation::Add:
            value_arithmetic_add(trait, ltype, lvalue, rtype, rvalue);
            break;
        case ArithmeticOperation::Sub:
            value_arithmetic_subtract(trait, ltype, lvalue, rtype, rvalue);
            break;
        case ArithmeticOperation::Mul:
            value_arithmetic_multiply(trait, ltype, lvalue, rtype, rvalue);
            break;
        case ArithmeticOperation::Div:
            value_arithmetic_divide(trait, ltype, lvalue, rtype, rvalue);
            break;
        default: RPY_UNREACHABLE();
    }
}

Value generics::dtl::value_arithmetic(
        ArithmeticOperation operation,
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    const auto common_type = compute_promotion(ltype, rtype);
    RPY_CHECK(common_type);

    const auto* trait = arithmetic_trait(common_type.get());
    RPY_CHECK_NE(trait, nullptr);
    RPY_CHECK(trait->has_operation(operation));

    Value result(common_type);
    result = ConstRef(ltype, lvalue);

    switch (operation) {
        case ArithmeticOperation::Add:
            value_arithmetic_add(
                    trait,
                    common_type.get(),
                    result.data(),
                    rtype,
                    rvalue
            );
            break;
        case ArithmeticOperation::Sub:
            value_arithmetic_subtract(
                    trait,
                    common_type.get(),
                    result.data(),
                    rtype,
                    rvalue
            );
            break;
        case ArithmeticOperation::Mul:
            value_arithmetic_multiply(
                    trait,
                    common_type.get(),
                    result.data(),
                    rtype,
                    rvalue
            );
            break;
        case ArithmeticOperation::Div:
            value_arithmetic_divide(
                    trait,
                    common_type.get(),
                    result.data(),
                    rtype,
                    rvalue
            );
            break;
        default: RPY_UNREACHABLE();
    }

    return result;
}
