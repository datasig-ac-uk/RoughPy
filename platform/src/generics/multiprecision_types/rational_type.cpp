//
// Created by sammorley on 25/11/24.
//

#include "rational_type.h"

#include <cstdio>

#include <gmp.h>

#include <roughpy/platform/alloc.h>

#include "mpz_hash.h"

using namespace rpy;
using namespace rpy::generics;

RationalType::RationalType()
    : m_arithmetic(this),
      m_comparison(this),
      m_number(this)
{}

void RationalType::inc_ref() const noexcept
{
    // Do Nothing
}
bool RationalType::dec_ref() const noexcept
{
    // do nothing
    return false;
}
intptr_t RationalType::ref_count() const noexcept { return 1; }
const std::type_info& RationalType::type_info() const noexcept
{
    return typeid(mpq_t);
}
BasicProperties RationalType::basic_properties() const noexcept
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
            false};
}
size_t RationalType::object_size() const noexcept { return sizeof(mpq_ptr); }
string_view RationalType::name() const noexcept { return "rational"; }
string_view RationalType::id() const noexcept { return "apr"; }
void* RationalType::allocate_object() const
{
    auto* new_rat = static_cast<mpq_ptr>(
            mem::aligned_alloc(alignof(mpq_t), sizeof(mpq_t))
    );

    if (new_rat == nullptr) { throw std::bad_alloc(); }

    mpq_init(new_rat);
    return new_rat;
}
void RationalType::free_object(void* ptr) const
{
    RPY_DBG_ASSERT_NE(ptr, nullptr);
    mpq_clear(static_cast<mpq_ptr>(ptr));
    mem::aligned_free(ptr);
}
bool RationalType::parse_from_string(void* data, string_view str) const noexcept
{
    RPY_DBG_ASSERT_NE(data, nullptr);

    auto* ptr = static_cast<mpq_ptr>(data);

    if (mpq_set_str(ptr, str.data(), 10) == -1) { return false; }

    // Just in case the string was not a canonical rational number, reduce it
    mpq_canonicalize(ptr);

    return true;
}
void RationalType::copy_or_move(
        void* dst,
        const void* src,
        size_t count,
        bool move
) const
{
    RPY_DBG_ASSERT_NE(dst, nullptr);

    if (count == 0) { return; }

    auto* dst_ptr = static_cast<mpq_ptr>(dst);
    if (src == nullptr) {
        for (size_t i = 0; i < count; ++i) { mpq_set_si(++dst_ptr, 0, 1); }
    } else if (move) {
        auto* src_ptr = static_cast<mpq_ptr>(const_cast<void*>(src));
        for (size_t i = 0; i < count; ++i) {
            mpq_swap(++dst_ptr, ++src_ptr);
            mpq_clear(src_ptr);
        }
    } else {
        const auto* src_ptr = static_cast<mpq_srcptr>(src);
        for (size_t i = 0; i < count; ++i) { mpq_set(++dst_ptr, ++src_ptr); }
    }
}
void RationalType::destroy_range(void* data, size_t count) const
{
    RPY_DBG_ASSERT_NE(data, nullptr);
    auto* ptr = static_cast<mpq_ptr>(data);
    for (size_t i = 0; i < count; ++i) { mpq_clear(ptr++); }
}
std::unique_ptr<const ConversionTrait>
RationalType::convert_to(const Type& type) const noexcept
{
    return Type::convert_to(type);
}
std::unique_ptr<const ConversionTrait>
RationalType::convert_from(const Type& type) const noexcept
{
    return Type::convert_from(type);
}
const BuiltinTrait*
RationalType::get_builtin_trait(BuiltinTraitID id) const noexcept
{
    switch (id) {
        case BuiltinTraitID::Comparison:
            return &m_comparison;
        case BuiltinTraitID::Arithmetic:
            return &m_arithmetic;
        case BuiltinTraitID::Number:
            return &m_number;
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}
const std::ostream&
RationalType::display(std::ostream& os, const void* value) const
{
    if (value == nullptr) { return os << 0; }

    const auto* rat = static_cast<mpq_srcptr>(value);

    // The GMP docs describe the size of a mpq string representation in the
    // documentation https://gmplib.org/manual/Rational-Conversions
    auto num_size = mpz_sizeinbase(mpq_numref(rat), 10);
    auto denom_size = mpz_sizeinbase(mpq_denref(rat), 10);

    string buffer;
    buffer.resize(num_size + denom_size + 3);


    mpq_get_str(buffer.data(), 10, rat);

    // The buffer has at least one null byte at the end, cut these off
    while (buffer.back() == '\0') { buffer.pop_back(); }
    return os << buffer;
}

hash_t RationalType::hash_of(const void* value) const noexcept
{
    if (value == nullptr) { return 0; }

    auto* rat = static_cast<mpq_srcptr>(value);
    auto num_hash = mpz_hash(mpq_numref(rat));
    const auto denom_hash = mpz_hash(mpq_denref(rat));

    hash_combine(num_hash, denom_hash);
    return num_hash;
}

TypePtr RationalType::get() noexcept
{
    static RationalType tp;
    return &tp;
}
