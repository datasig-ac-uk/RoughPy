//
// Created by sam on 14/11/23.
//

#include "raw_bytes.h"
#include "do_macro.h"

#include "builtin_scalars.h"
#include "types/monomial.h"

#include <random>
#include <sstream>

using namespace rpy;
using namespace scalars;

namespace {

template <typename T>
enable_if_t<is_trivially_copyable_v<T>>
to_raw_bytes_impl(std::vector<byte>& out, const T* data, dimn_t size)
{
    const auto nbytes = size * sizeof(T);
    const auto curr_size = out.size();
    out.resize(curr_size + nbytes);
    std::memcpy(out.data() + curr_size, data, nbytes);
}

void to_raw_bytes_impl(
        std::vector<byte>& out,
        const ArbitraryPrecisionRational* data,
        dimn_t count
)
{
    auto serial_int = [&out](auto* value) {
        out.push_back(static_cast<bool>(mpz_sgn(value)));
        const auto size
                = static_cast<uint64_t>(mpz_size(value)) * sizeof(mp_limb_t);
        auto it = out.insert(
                out.end(),
                reinterpret_cast<const byte*>(&size),
                reinterpret_cast<const byte*>((&size) + 1)
        );
        const auto* limb_ptr
                = reinterpret_cast<const byte*>(mpz_limbs_read(value));
        it = out.insert(it, limb_ptr, limb_ptr + size);
        (void) it;
    };

    out.reserve(out.size() + count * sizeof(ArbitraryPrecisionRational));
    for (dimn_t i = 0; i < count; ++i) {
        const auto& backend = data->backend();
        serial_int(mpq_numref(backend.data()));
        serial_int(mpq_denref(backend.data()));
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

void to_raw_bytes_impl(std::vector<byte>& out, const Monomial& value)
{
    uint64_t size = value.type();
    to_raw_bytes_impl(out, &size, 1);

    for (auto&& term : value) {
        to_raw_bytes_impl(out, term.first);
        to_raw_bytes_impl(out, &term.second, 1);
    }
}

void to_raw_bytes_impl_single(std::vector<byte>& out, const APPolyRat& value)
{
    uint64_t size = value.size();
    to_raw_bytes_impl(out, &size, 1);
    for (auto&& item : value) {
        to_raw_bytes_impl(out, item.first);
        to_raw_bytes_impl(out, &item.second, 1);
    }
}

void to_raw_bytes_impl(
        std::vector<byte>& out,
        const APPolyRat* data,
        dimn_t count
)
{
    for (dimn_t i = 0; i < count; ++i) {
        to_raw_bytes_impl_single(out, data[i]);
    }
}

}// namespace

std::vector<byte>
scalars::dtl::to_raw_bytes(const void* ptr, dimn_t size, PackedScalarType info)
{
    auto tp_info = type_info_from(info);
    std::vector<byte> out;
    // Reserve approximately the right amount of space to cut down number of
    // reallocations
    out.reserve(size * tp_info.bytes);
#define X(TP)                                                                  \
    to_raw_bytes_impl(out, (const TP*) ptr, size);                             \
    break
    DO_FOR_EACH_X(tp_info)
#undef X
    return out;
}

namespace {

template <typename T>
enable_if_t<is_trivially_copyable_v<T>>
from_raw_bytes_impl(T* dst, dimn_t count, Slice<const byte> bytes)
{
    const auto nbytes = count * sizeof(T);
    RPY_CHECK(bytes.size() >= nbytes);
    std::memcpy(dst, bytes.begin(), nbytes);
}

void from_raw_bytes_impl(
        ArbitraryPrecisionRational* dst,
        dimn_t count,
        Slice<const byte> bytes,
        dimn_t* final_offset = nullptr
)
{

    const auto* src = bytes.data();
    dimn_t remaining = bytes.size();

    auto read_int = [&remaining, &src](auto* value) {
        RPY_CHECK(remaining >= 1 + sizeof(uint64_t));
        bool is_negative = static_cast<bool>(src++);
        remaining -= 1;
        uint64_t nbytes;
        std::memcpy(&nbytes, src, sizeof(uint64_t));
        src += nbytes;
        remaining -= sizeof(uint64_t);

        RPY_CHECK(remaining >= nbytes);
        auto n_limbs = (nbytes + sizeof(mp_limb_t) - 1) / sizeof(mp_limb_t);
        std::memcpy(mpz_limbs_write(value, n_limbs), src, nbytes);

        auto size = is_negative ? -static_cast<mp_size_t>(n_limbs)
                                : static_cast<mp_size_t>(n_limbs);

        mpz_limbs_finish(value, size);

        src += nbytes;
        remaining -= nbytes;
    };

    for (dimn_t i = 0; i < count; ++i) {
        auto& backend = dst[i].backend();
        read_int(mpq_numref(backend.data()));
        read_int(mpq_denref(backend.data()));
    }

    if (final_offset != nullptr) {
        *final_offset = static_cast<dimn_t>(src - bytes.begin());
    }
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
    from_raw_bytes_impl(
            &integral,
            1,
            {src + sizeof(packed_type), sizeof(integral_type)}
    );
    value = indeterminate_type(packed, integral);
}

dimn_t from_raw_bytes_impl(Monomial& value, Slice<const byte> bytes)
{
    constexpr auto sizeof_idep = sizeof(indeterminate_type);
    constexpr auto sizeof_term = sizeof_idep + sizeof(deg_t);
    RPY_CHECK(bytes.size() > sizeof(uint64_t));
    uint64_t size_large = 0;
    from_raw_bytes_impl(&size_large, 1, bytes);
    auto size = static_cast<dimn_t>(size_large);

    RPY_CHECK(bytes.size() > size_large * sizeof_term + sizeof(uint64_t));
    const auto* src = bytes.data() + sizeof(uint64_t);

    indeterminate_type tmp_idep(0, 0);
    deg_t power = 0;
    for (dimn_t i = 0; i < size; ++i) {
        // TODO: This isn't right. Indeterminates are stored as 5 bytes, not 4
        from_raw_bytes_impl(tmp_idep, {src, sizeof_idep});
        from_raw_bytes_impl(&power, 1, {src + sizeof_idep, sizeof(deg_t)});
        // value[tmp_idep] = power;
        src += sizeof_term;
    }

    return size + sizeof(uint64_t);
}

dimn_t from_raw_bytes_impl_single(APPolyRat& value, Slice<const byte> bytes)
{
    RPY_CHECK(bytes.size() >= sizeof(uint64_t));
    uint64_t count_large = 0;
    from_raw_bytes_impl(&count_large, 1, bytes);
    dimn_t final_offset = sizeof(uint64_t);

    Monomial tmp_key;
    ArbitraryPrecisionRational tmp_value;

    const auto* src = bytes.data() + sizeof(uint64_t);
    auto count = static_cast<dimn_t>(count_large);
    dimn_t advance = 0;
    dimn_t offset = 0;
    dimn_t remaining = bytes.size() - sizeof(uint64_t);

    for (dimn_t i = 0; i < count; ++i) {
        advance = from_raw_bytes_impl(tmp_key, {src, remaining});
        from_raw_bytes_impl(
                &tmp_value,
                1,
                {src + advance, remaining - advance},
                &offset
        );
        advance += offset;
        remaining -= advance;
        src += advance;
        final_offset += advance;

        value[tmp_key] = tmp_value;
    }

    return final_offset;
}

void from_raw_bytes_impl(APPolyRat* dst, dimn_t count, Slice<const byte> bytes)
{
    const auto* src = bytes.data();
    auto remaining = bytes.size();
    dimn_t advance = 0;

    for (dimn_t i = 0; i < count; ++i) {
        advance = from_raw_bytes_impl_single(dst[i], {src, remaining});
        src += advance;
        remaining -= advance;
    }
}

}// namespace

void scalars::dtl::from_raw_bytes(
        void* dst,
        dimn_t count,
        Slice<byte> bytes,
        PackedScalarType info
)
{
#define X(TP) return from_raw_bytes_impl((TP*) dst, count, bytes)
    DO_FOR_EACH_X(type_info_from(info))
#undef X
}
