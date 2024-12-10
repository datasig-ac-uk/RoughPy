//
// Created by sam on 27/11/24.
//

#include "polynomial_type.h"

#include <algorithm>
#include <typeinfo>
#include <utility>

#include <ctre.hpp>


#include <roughpy/core/construct_inplace.h>
#include <roughpy/core/debug_assertion.h>

#include <roughpy/platform/alloc.h>


#include "indeterminate.h"
#include "monomial.h"
#include "polynomial.h"
#include "polynomial_conversion.h"
#include "conversion_helpers.h"
#include "generics/builtin_types/conversion_factory.h"


using namespace rpy;
using namespace rpy::generics;

PolynomialType::PolynomialType()
    : m_arithmetic(this, MultiPrecisionTypes::get().rational_type.get()),
      m_comparison(this),
      m_number(this) {}

void PolynomialType::inc_ref() const noexcept
{
    // do nothing
}

bool PolynomialType::dec_ref() const noexcept
{
    // Do nothing
    return false;
}

intptr_t PolynomialType::ref_count() const noexcept { return 1; }

const std::type_info& PolynomialType::type_info() const noexcept
{
    return typeid(Polynomial);
}

BasicProperties PolynomialType::basic_properties() const noexcept
{
    return {false, false, false, false, false, false, false, false, true,
            false, false};
}

size_t PolynomialType::object_size() const noexcept
{
    return sizeof(Polynomial);
}

string_view PolynomialType::name() const noexcept { return "Polynomial"; }

string_view PolynomialType::id() const noexcept { return "poly"; }

void* PolynomialType::allocate_object() const
{
    auto* ptr = static_cast<Polynomial*>(mem::small_object_alloc(
        sizeof(Polynomial)));
    if (ptr == nullptr) { throw std::bad_alloc(); }
    construct_inplace(ptr);

    return ptr;
}

void PolynomialType::free_object(void* ptr) const
{
    auto* poly = static_cast<Polynomial*>(ptr);
    poly->~Polynomial();
    mem::small_object_free(poly, sizeof(Polynomial));
}


void PolynomialType::copy_or_fill(void* dst,
                                  const void* src,
                                  size_t count,
                                  bool uninit) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);
    auto* dst_poly = static_cast<Polynomial*>(dst);
    const auto* src_poly = static_cast<const Polynomial*>(src);

    if (src_poly == nullptr) {
        if (uninit) {
            std::uninitialized_default_construct_n(dst_poly, count);
        } else {
            std::fill_n(dst_poly, count, Polynomial());
        }
    } else {
        if (uninit) {
            std::uninitialized_copy_n(src_poly, count, dst_poly);
        } else {
            std::copy_n(src_poly, count, dst_poly);
        }
    }
}

void PolynomialType::move(void* dst, void* src, size_t count, bool uninit) const
{
    if (RPY_UNLIKELY(count == 0)) { return; }

    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_poly = static_cast<Polynomial*>(dst);
    auto* src_poly = static_cast<const Polynomial*>(src);

    if (uninit) {
        std::uninitialized_move_n(src_poly, count, dst_poly);
    } else {
        std::copy_n(std::make_move_iterator(src_poly), count, dst_poly);
    }
}

void PolynomialType::destroy_range(void* data, size_t count) const
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* poly = static_cast<Polynomial*>(data);
    std::destroy_n(poly, count);
}

std::unique_ptr<const ConversionTrait> PolynomialType::
convert_to(const Type& type) const noexcept
{
    static const auto table = conv::make_poly_conversion_to_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(&type, this);
    }

    return Type::convert_to(type);
}

std::unique_ptr<const ConversionTrait> PolynomialType::
convert_from(const Type& type) const noexcept
{
    static const auto table = conv::make_poly_conversion_from_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(&type, this);
    }

    return Type::convert_from(type);
}

const BuiltinTrait* PolynomialType::
get_builtin_trait(BuiltinTraitID id) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Arithmetic: return &m_arithmetic;
        case BuiltinTraitID::Comparison: return &m_comparison;
        case BuiltinTraitID::Number: return &m_number;
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

const std::ostream& PolynomialType::display(std::ostream& os,
                                            const void* value) const
{
    if (value == nullptr) { return os << "{ }"; }
    const auto* poly = static_cast<const Polynomial*>(value);
    poly_print(os, *poly);
    return os;
}

hash_t PolynomialType::hash_of(const void* value) const noexcept
{
    if (value == nullptr) { return 0; }
    return hash_value(*static_cast<const Polynomial*>(value));
}

TypePtr PolynomialType::get() noexcept
{
    static PolynomialType type;
    return &type;
}


/******************************************************************************
 *                                 Parser                                     *
 ******************************************************************************/

namespace {


template <typename T>
bool parse_integer(T& value, string_view str) noexcept
{

    auto parse_result = std::from_chars(
        str.data(),
        str.data() + str.size(),
        value,
        10
    );

    return parse_result.ec == std::errc();
}

bool parse_dbl_coeff(generics::dtl::RationalCoeff& coeff, bool negative, string_view data) noexcept
{
    try {
        auto dbl_coeff = std::stod(string(data));
        mpq_set_d(coeff.content, negative ? -dbl_coeff : dbl_coeff);
    } catch (std::invalid_argument&) {
        return false;
    }
    return true;
}

bool parse_rat_coeff(generics::dtl::RationalCoeff& coeff, bool neg, string_view data) noexcept
{
    int64_t numerator;
    int64_t denominator = 1;

    constexpr auto rat_pattern = ctll::fixed_string{
        R"((?<num>[1-9]\d*)(?:\/(?<den>[1-9]\d*))?)"
    };

    if (auto match = ctre::match<rat_pattern>(data)) {
        if (auto num = match.get<"num">()) {
            if (!parse_integer(numerator, num.view())) {
                return false;
            }
        }
        if (auto den = match.get<"den">()) {
            if (!parse_integer(denominator, den.view())) {
                return false;
            }
        }
    } else {
        return false;
    }

    mpq_set_si(coeff.content, neg ? -numerator : numerator, denominator);

    return true;
}


bool parse_monomial(Monomial& result, string_view monomial_string) noexcept
{
    constexpr auto monomial_pattern = ctll::fixed_string{
            R"(([a-zA-Z])([1-9]\d*)(?:\^([1-9]\d*))?)"
    };

    for (auto match : ctre::search_all<monomial_pattern>(monomial_string)) {
        auto prefix = match.get<1>();
        RPY_DBG_ASSERT(prefix);
        auto index = match.get<2>();
        RPY_DBG_ASSERT(index);

        uint64_t index_digits = 0;
        if (!parse_integer(index_digits, index.view())) {
            result.clear();
            return false;
        }

        Indeterminate ind(*prefix.begin(), index_digits);

        deg_t pow = 1;
        if (auto power = match.get<3>()) {
            if (!parse_integer(pow, power.view())) {
                result.clear();
                return false;
            }
        }

        result.emplace(ind, pow);
    }

    return true;
}

}

bool PolynomialType::parse_from_string(void* data,
                                       string_view str) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);

    auto* poly = static_cast<Polynomial*>(data);

    constexpr auto term_pattern = ctll::fixed_string{
            R"((?<sgn>\+|\-)?(?:(?<dbl>\d+\.\d*)|(?<rat>[1-9]\d*(?:\/[1-9]\d*)?))(?:\((?<mon>(?:[a-zA-Z][1-9]\d*(?:\^(?:[1-9]\d*))?)+)\))?)"
    };

    generics::dtl::RationalCoeff tmp_coeff;
    for (auto match : ctre::search_all<term_pattern>(str)) {
        auto sign_m = match.get<"sgn">();
        bool negative = sign_m && *sign_m.begin() == '-';

        if (auto dbl_grp = match.get<"dbl">()) {
            if (!parse_dbl_coeff(tmp_coeff, negative, dbl_grp.view())) {
                poly->clear();
                return false;
            }
        } else if (auto rat_grp = match.get<"rat">()) {
            if (!parse_rat_coeff(tmp_coeff, negative, rat_grp.view())) {
                poly->clear();
                return false;
            }
        }

        Monomial mon;
        if (auto monomial_grp = match.get<"mon">()) {
            if (!parse_monomial(mon, monomial_grp.view())) {
                poly->clear();
                return false;
            }
        }

        poly->emplace(mon, std::move(tmp_coeff));
    }

    return true;
}