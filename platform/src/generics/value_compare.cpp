//
// Created by sam on 15/11/24.
//

#include "roughpy/generics/values.h"

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"

#include "roughpy/generics/comparison_trait.h"

using namespace rpy;
using namespace rpy::generics;

namespace {

template <typename Fn>
bool value_compare_converting(
    Fn&& fn, const Type* ltype, const void* lvalue, const Type* rtype, const void* rvalue)
{
    if (RPY_LIKELY(ltype == rtype)) {
        const auto* trait = trait_cast<ComparisonTrait>(
            ltype->get_builtin_trait(BuiltinTraitID::Comparison));

        return fn(trait, lvalue, rvalue);
    }

    const auto common_tp = compute_promotion(ltype, rtype);
    RPY_CHECK_NE(common_tp, nullptr);
    const auto* trait = trait_cast<ComparisonTrait>(
        ltype->get_builtin_trait(BuiltinTraitID::Comparison));

    Value tmp(common_tp);
    if (ltype == common_tp) {
        tmp = ConstRef(rtype, rvalue);
        rvalue = tmp.data();
    } else {
        tmp = ConstRef(ltype, lvalue);
        lvalue = tmp.data();
    }

    return fn(trait, lvalue, rvalue);
}

bool value_compare_equal(
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    if (ltype == nullptr && rtype == nullptr) {
        return true;
    }
    RPY_CHECK(ltype != nullptr && rtype != nullptr);

    return value_compare_converting(
        [](const ComparisonTrait* trait, const void* lptr, const void* rptr) {
            return trait->unsafe_compare_equal(lptr, rptr);
        },
        ltype,
        lvalue,
        rtype,
        rvalue
    );
}

void check_types(const Type* ltype, const Type* rtype)
{
    RPY_CHECK_NE(ltype, nullptr);
    RPY_CHECK_NE(rtype, nullptr);

}

bool value_compare_less(
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    check_types(ltype, rtype);

    return value_compare_converting(
        [](const ComparisonTrait* trait, const void* lptr, const void* rptr) {
            return trait->unsafe_compare_less(lptr, rptr);
        },
        ltype,
        lvalue,
        rtype,
        rvalue
    );

}

bool value_compare_less_equal(
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    check_types(ltype, rtype);

    return value_compare_converting(
        [](const ComparisonTrait* trait, const void* lptr, const void* rptr) {
            return trait->unsafe_compare_less_equal(lptr, rptr);
        },
        ltype,
        lvalue,
        rtype,
        rvalue
    );
}

bool value_compare_greater(
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    check_types(ltype, rtype);

    return value_compare_converting(
        [](const ComparisonTrait* trait, const void* lptr, const void* rptr) {
            return trait->unsafe_compare_greater(lptr, rptr);
        },
        ltype,
        lvalue,
        rtype,
        rvalue
    );
}

bool value_compare_greater_equal(
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    return value_compare_converting(
        [](const ComparisonTrait* trait, const void* lptr, const void* rptr) {
            return trait->unsafe_compare_greater_equal(lptr, rptr);
        },
        ltype,
        lvalue,
        rtype,
        rvalue
    );
}

}// namespace

bool rpy::generics::dtl::values_compare(
        ComparisonType comp,
        const Type* ltype,
        const void* lvalue,
        const Type* rtype,
        const void* rvalue
)
{
    switch (comp) {
        case ComparisonType::Equal:
            return value_compare_equal(ltype, lvalue, rtype, rvalue);
        case ComparisonType::Less:
            return value_compare_less(ltype, lvalue, rtype, rvalue);
        case ComparisonType::LessEqual:
            return value_compare_less_equal(ltype, lvalue, rtype, rvalue);
        case ComparisonType::Greater:
            return value_compare_greater(ltype, lvalue, rtype, rvalue);
        case ComparisonType::GreaterEqual:
            return value_compare_greater_equal(ltype, lvalue, rtype, rvalue);
    }
    RPY_UNREACHABLE_RETURN(false);
}
