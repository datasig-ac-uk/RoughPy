//
// Created by sam on 14/11/23.
//


#include "raw_bytes.h"

#include "roughpy/core/check.h"  // for throw_exception

#include "do_macro.h"

#include "scalar_types.h"
#include "scalar_serialization.h"

#include <random>
#include <sstream>

using namespace rpy;
using namespace scalars;


namespace {

template <typename T>
enable_if_t<is_trivially_copyable_v<T>> to_raw_bytes_impl(
    std::vector<byte>& out,
    const T* data,
    dimn_t size)
{
    const auto nbytes = size * sizeof(T);
    const auto curr_size = out.size();
    out.resize(curr_size + nbytes);
    std::memcpy(out.data() + curr_size, data, nbytes);
}

template <typename I>
void to_raw_bytes_impl_helper(std::vector<byte>& out,
                              scalars::dtl::MPIntegerSerializationHelper<I>&
                              helper)
{
    const auto curr_size = out.size();
    out.resize(curr_size + helper.total_bytes());

    auto it = out.begin() + curr_size;
    *it = (helper.is_negative()) ? 1 : 0;
    ++it;
    uint64_t limbs_size = helper.size();
    it = std::copy_n(reinterpret_cast<const byte*>(&limbs_size),
                     sizeof(uint8_t),
                     it);
    std::copy_n(reinterpret_cast<const byte*>(helper.limbs()), limbs_size, it);
}


void to_raw_bytes_impl(std::vector<byte>& out,
                       const rational_scalar_type* data,
                       dimn_t count)
{
    out.reserve(out.size() + count * sizeof(rational_scalar_type));
    for (dimn_t i = 0; i < count; ++i) {
        const auto& backend = data->backend();
#if RPY_USING_GMP
        using helper_t = scalars::dtl::MPIntegerSerializationHelper<
            remove_pointer_t
            <mpz_srcptr>>;

        helper_t num_helper(mpq_numref(backend.data()));
        helper_t den_helper(mpq_denref(backend.data()));
#else
    using helper_t = scalars::dtl::MPIntegerSerializationHelper<const boost::multiprecision::cpp_int_backend>;

    helper_t num_helper(backend.num());
    helper_t den_helper(backend.den());
#endif

        to_raw_bytes_impl_helper(out, num_helper);
        to_raw_bytes_impl_helper(out, den_helper);
    }
}

void to_raw_bytes_impl(std::vector<byte>& out, const indeterminate_type& value)
{
    using packed_type = typename indeterminate_type::packed_type;
    using integral_type = typename indeterminate_type::integral_type;

    auto packed = static_cast<packed_type>(value);
    auto integral = static_cast<integral_type>(value);

    to_raw_bytes_impl(out, &packed, 1);
    to_raw_bytes_impl(out, &integral, 1);
}

void to_raw_bytes_impl(std::vector<byte>& out, const monomial& value)
{
    uint64_t size = value.type();
    to_raw_bytes_impl(out, &size, 1);

    for (auto&& term : value) {
        to_raw_bytes_impl(out, term.first);
        to_raw_bytes_impl(out, &term.second, 1);
    }
}

void to_raw_bytes_impl_single(std::vector<byte>& out,
                              const rational_poly_scalar& value)
{
    uint64_t size = value.size();
    to_raw_bytes_impl(out, &size, 1);
    for (auto&& term : value) {
        to_raw_bytes_impl(out, term.key());
        to_raw_bytes_impl(out, &term.value(), 1);
    }
}

void to_raw_bytes_impl(std::vector<byte>& out,
                       const rational_poly_scalar* data,
                       dimn_t count)
{
    for (dimn_t i = 0; i < count; ++i) {
        to_raw_bytes_impl_single(out, data[i]);
    }
}


}


std::vector<byte> scalars::dtl::to_raw_bytes(
    const void* ptr,
    dimn_t size,
    const devices::TypeInfo& info
)
{
    std::vector<byte> out;
    // Reserve approximately the right amount of space to cut down number of
    // reallocations
    out.reserve(size * info.bytes);
#define X(TP) to_raw_bytes_impl(out, (const TP*) ptr, size); break
    DO_FOR_EACH_X(info)
#undef X
    return out;
}


namespace {

template <typename T>
enable_if_t<is_trivially_copyable_v<T>> from_raw_bytes_impl(
    T* dst,
    dimn_t count,
    Slice<const byte> bytes)
{
    const auto nbytes = count * sizeof(T);
    RPY_CHECK(bytes.size() >= nbytes);
    std::memcpy(dst, bytes.begin(), nbytes);
}

template <typename I>
dimn_t from_raw_bytes_impl(
    scalars::dtl::MPIntegerSerializationHelper<I>& helper,
    Slice<const byte> bytes)
{
    auto advance = 1 + sizeof(uint64_t);
    const auto* src = bytes.data();
    RPY_CHECK(bytes.size() > advance);

    auto is_negative = static_cast<bool>(*(src++));
    uint64_t nbytes_large = 0;
    std::memcpy(&nbytes_large, src, sizeof(uint64_t));
    src += sizeof(uint64_t);
    auto nbytes = static_cast<dimn_t>(nbytes_large);
    advance += nbytes;
    RPY_CHECK(bytes.size() > advance);

    dimn_t n_limbs = 0;
    if (nbytes > 0) {
        n_limbs = round_up_divide(nbytes, helper.sizeof_limb());
        std::memcpy(helper.resize(n_limbs), src, nbytes);
    }
    helper.finalize(n_limbs, is_negative);

    return advance;
}

void from_raw_bytes_impl(rational_scalar_type* dst,
                         dimn_t count,
                         Slice<const byte> bytes,
                         dimn_t* final_offset = nullptr)
{
#if RPY_USING_GMP
    using helper_t = scalars::dtl::MPIntegerSerializationHelper<remove_pointer_t
        <mpz_ptr>>;
#else
    using helper_t = scalars::dtl::MPIntegerSerializationHelper<boost::multiprecision::cpp_int_backend>;
#endif

    const auto* src = bytes.data();
    dimn_t offset = 0;
    dimn_t remaining = bytes.size();
    for (dimn_t i = 0; i < count; ++i) {
        auto& backend = dst[i].backend();

#if RPY_USING_GMP
        helper_t num_helper(mpq_numref(backend.data()));
        helper_t den_helper(mpq_denref(backend.data()));
#else
        helper_t num_helper(backend.num());
        helper_t den_helper(backend.den());
#endif

        auto advance = from_raw_bytes_impl(num_helper, {src, remaining});
        src += advance;
        remaining -= advance;
        advance = from_raw_bytes_impl(den_helper, {src, remaining});
        src += advance;
        remaining -= advance;

        offset += advance;
    }

    if (final_offset != nullptr) { *final_offset = offset; }
}

void from_raw_bytes_impl(indeterminate_type& value, Slice<const byte> bytes)
{
    using packed_type = typename indeterminate_type::packed_type;
    using integral_type = typename indeterminate_type::integral_type;

    RPY_CHECK(bytes.size() >= sizeof(packed_type) + sizeof(integral_type));

    const auto* src = bytes.data();
    packed_type packed;
    from_raw_bytes_impl(&packed, 1, {src, sizeof(packed_type)});
    integral_type integral;
    from_raw_bytes_impl(&integral,
                        1,
                        {src + sizeof(packed_type), sizeof(integral_type)});
    value = indeterminate_type(packed, integral);
}

dimn_t from_raw_bytes_impl(monomial& value, Slice<const byte> bytes)
{
    constexpr auto sizeof_idep = sizeof(typename indeterminate_type::packed_type) + sizeof(typename indeterminate_type::integral_type);
    constexpr auto sizeof_term = sizeof_idep + sizeof(lal::deg_t);
    RPY_CHECK(bytes.size() > sizeof(uint64_t));
    uint64_t size_large = 0;
    from_raw_bytes_impl(&size_large, 1, bytes);
    auto size = static_cast<dimn_t>(size_large);

    RPY_CHECK(bytes.size() > size_large*sizeof_term + sizeof(uint64_t));
    const auto* src = bytes.data() + sizeof(uint64_t);

    indeterminate_type tmp_idep(0, 0);
    lal::deg_t power = 0;
    for (dimn_t i=0; i<size; ++i) {
        from_raw_bytes_impl(tmp_idep, {src, sizeof_idep});
        from_raw_bytes_impl(&power, 1, {src + sizeof_idep, sizeof(lal::deg_t)});
        value[tmp_idep] = power;
        src += sizeof_term;
    }

    return size + sizeof(uint64_t);
}

dimn_t from_raw_bytes_impl_single(rational_poly_scalar& value, Slice<const byte> bytes)
{
    RPY_CHECK(bytes.size() >= sizeof(uint64_t));
    uint64_t count_large = 0;
    from_raw_bytes_impl(&count_large, 1, bytes);
    dimn_t final_offset = sizeof(uint64_t);

    monomial tmp_key;
    rational_scalar_type tmp_value;

    const auto* src = bytes.data() + sizeof(uint64_t);
    auto count = static_cast<dimn_t>(count_large);
    dimn_t advance = 0;
    dimn_t offset = 0;
    dimn_t remaining = bytes.size() - sizeof(uint64_t);

    for (dimn_t i=0; i<count; ++i) {
        advance = from_raw_bytes_impl(tmp_key, {src, remaining});
        from_raw_bytes_impl(&tmp_value, 1, {src + advance, remaining-advance}, &offset);
        advance += offset;
        remaining -= advance;
        src += advance;
        final_offset += advance;

        value[tmp_key] = tmp_value;
    }

    return final_offset;
}


void from_raw_bytes_impl(rational_poly_scalar* dst, dimn_t count, Slice<const byte> bytes)
{
    const auto* src = bytes.data();
    auto remaining = bytes.size();
    dimn_t advance = 0;

    for (dimn_t i=0; i<count; ++i) {
        advance = from_raw_bytes_impl_single(dst[i], {src, remaining});
        src += advance;
        remaining -= advance;
    }

}



}


void scalars::dtl::from_raw_bytes(
    void* dst,
    dimn_t count,
    Slice<byte> bytes,
    const devices::TypeInfo& info
)
{
#define X(TP) return from_raw_bytes_impl((TP*) dst, count, bytes)
    DO_FOR_EACH_X(info)
#undef X
}
