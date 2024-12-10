//
// Created by sammorley on 25/11/24.
//

#include "integer_type.h"

#include "mpz_hash.h"

#include <gmp.h>


#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/hash.h"
#include "roughpy/core/types.h"

#include "integer_conversion.h"

using namespace rpy;
using namespace rpy::generics;

IntegerType::IntegerType()
    : m_arithmetic(this),
      m_comparison(this),
      m_number(this) {}

void IntegerType::inc_ref() const noexcept
{
    // Do Nothing
}

bool IntegerType::dec_ref() const noexcept
{
    // do nothing
    return false;
}

intptr_t IntegerType::ref_count() const noexcept { return 1; }

const std::type_info& IntegerType::type_info() const noexcept
{
    return typeid(mpz_t);
}

BasicProperties IntegerType::basic_properties() const noexcept
{
    return {false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            false
    };
}

size_t IntegerType::object_size() const noexcept { return sizeof(mpz_t); }

string_view IntegerType::name() const noexcept
{
    return "MultiPrecisionInteger";
}

string_view IntegerType::id() const noexcept { return "apz"; }

void* IntegerType::allocate_object() const
{
    auto* new_int = static_cast<mpz_ptr>(
        mem::aligned_alloc(alignof(mpz_t), sizeof(mpz_t))
    );

    if (new_int == nullptr) { throw std::bad_alloc(); }

    mpz_init(new_int);
    return new_int;
}

void IntegerType::free_object(void* ptr) const
{
    RPY_DBG_ASSERT_NE(ptr, nullptr);
    mpz_clear(static_cast<mpz_ptr>(ptr));
    mem::aligned_free(ptr, sizeof(mpz_t));
}

bool IntegerType::parse_from_string(void* data, string_view str) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* ptr = static_cast<mpz_ptr>(data);
    string tmp(str);
    return mpz_set_str(ptr, tmp.c_str(), 10) != -1;
}

void IntegerType::copy_or_fill(
    void* dst,
    const void* src,
    size_t count,
    bool uninit
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);

    if (RPY_UNLIKELY(count == 0)) { return; }

    auto* dst_ptr = static_cast<mpz_ptr>(dst);
    if (src == nullptr) {
        if (uninit) {
            for (size_t i = 0; i < count; ++i, ++dst_ptr) {
                // mpz_init sets the value to 0
                mpz_init(dst_ptr);
            }
        } else {
            for (size_t i = 0; i < count; ++i, ++dst_ptr) {
                mpz_set_si(dst_ptr, 0);
            }
        }
    } else {
        const auto* src_ptr = static_cast<mpz_srcptr>(src);
        if (uninit) {
            for (size_t i = 0; i < count; ++i, ++src_ptr, ++dst_ptr) {
                mpz_init(dst_ptr);
                mpz_set(dst_ptr, src_ptr);
            }
        } else {
            for (size_t i = 0; i < count; ++i, ++src_ptr, ++dst_ptr) {
                mpz_set(dst_ptr, src_ptr);
            }
        }
    }
}

void IntegerType::move(void* dst, void* src, size_t count, bool uninit) const
{
    if (RPY_UNLIKELY(count == 0)) { return; }
    RPY_DBG_ASSERT_NE(dst, nullptr);
    RPY_DBG_ASSERT_NE(src, nullptr);

    auto* dst_ptr = static_cast<mpz_ptr>(dst);
    auto* src_ptr = static_cast<mpz_ptr>(src);
    if (uninit) {
        for (size_t i = 0; i < count; ++i, ++dst_ptr, ++src_ptr) {
            mpz_init(dst_ptr);
            mpz_swap(dst_ptr, src_ptr);
        }
    } else {
        for (size_t i = 0; i < count; ++i, ++dst_ptr, ++src_ptr) {
            mpz_swap(dst_ptr, src_ptr);
        }
    }
}

void IntegerType::destroy_range(void* data, size_t count) const
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* ptr = static_cast<mpz_ptr>(data);
    for (size_t i = 0; i < count; ++i) { mpz_clear(ptr++); }
}

std::unique_ptr<const ConversionTrait>
IntegerType::convert_to(const Type& type) const noexcept
{
    static const auto table = conv::make_mpint_conversion_to_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(this, &type);
    }

    return Type::convert_to(type);
}

std::unique_ptr<const ConversionTrait>
IntegerType::convert_from(const Type& type) const noexcept
{
    static const auto table = conv::make_mpint_conversion_from_table();
    constexpr Hash<string_view> hasher;

    auto it = table.find(hasher(type.id()));
    if (it != table.end()) {
        return it->second->make(this, &type);
    }

    return Type::convert_from(type);
}

const BuiltinTrait*
IntegerType::get_builtin_trait(BuiltinTraitID id) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Comparison: return &m_comparison;
        case BuiltinTraitID::Arithmetic: return &m_arithmetic;
        case BuiltinTraitID::Number: return &m_number;
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}

const std::ostream&
IntegerType::display(std::ostream& os, const void* value) const
{
    if (value == nullptr) { return os << 0; }

    const auto* ptr = static_cast<mpz_srcptr>(value);

    // The GMP docs describe the size of a mpz string representation in the
    // documentation https://gmplib.org/manual/Converting-Integers
    auto num_chars = mpz_sizeinbase(ptr, 10) + 2;

    string buffer;
    buffer.resize(num_chars);

    mpz_get_str(buffer.data(), 10, ptr);

    while (buffer.back() == '\0') { buffer.pop_back(); }
    return os << buffer;
}

hash_t IntegerType::hash_of(const void* value) const noexcept
{
    if (value == nullptr) { return 0; }

    const auto* ptr = static_cast<mpz_srcptr>(value);
    return mpz_hash(ptr);
}

TypePtr IntegerType::get() noexcept
{
    static IntegerType tp;
    return &tp;
}