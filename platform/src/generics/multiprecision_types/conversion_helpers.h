//
// Created by sammorley on 29/11/24.
//

#ifndef ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H
#define ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H


#include <limits>
#include <utility>

#include <gmp.h>
#include <mpfr.h>

#include "roughpy/core/types.h"
#include "roughpy/core/traits.h"

#include "generics/builtin_types/conversion_helpers.h"

#include "multiprecision_type_ids.h"

namespace rpy::generics::conv {


template <typename I>
struct ConversionHelper<MPInt, I, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = mpz_srcptr;
    using to_ptr = I*;
    
    static constexpr bool is_always_exact = false;

    static ConversionResult ConversionResult(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (is_signed_v<I>) {
            if constexpr (sizeof(I) < 2) {
                if (ensure_exact && !mpz_fits_sshort_p(src)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 4) {
                if (ensure_exact && !mpz_fits_sint_p(src)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 8) {
                if (ensure_exact && !mpz_fits_slong_p(src)) {
                    return ConversionResult::Inexact;
                }
            }

            *dst = mpz_get_si(src);
            return ConversionResult::Success;
        } else {
            // If I is unsigned, first check if the source is negative
            if (mpz_sgn(src) < 0) { return ConversionResult::Failed; }

            if constexpr (sizeof(I) < 2) {
                if (ensure_exact && !mpz_fits_ushort_p(src)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 4) {
                if (ensure_exact && !mpz_fits_uint_p(src)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 8) {
                if (ensure_exact && !mpz_fits_ulong_p(src)) {
                    return ConversionResult::Inexact;
                }
            }

            *dst = mpz_get_ui(src);
            return ConversionResult::Success;
        }

    }
};


template <typename I>
struct ConversionHelper<I, MPInt, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = const I*;
    using to_ptr = mpz_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        if constexpr (is_signed_v<I>) {
            mpz_set_si(dst, *src);
        } else {
            mpz_set_ui(dst, *src);
        }
        return ConversionResult::Success;
    }
};

template <typename F>
struct ConversionHelper<MPInt, F, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = mpz_srcptr;
    using to_ptr = const F*;

    static constexpr bool is_always_exact = false;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        if (ensure_exact) {
            int64_t tmp;
            auto first_result = ConversionHelpers<int64_t, F>::from(
                &tmp,
                src,
                ensure_exact);
            if (first_result != ConversionResult::Success) {
                return first_result;
            }
            return ConversionHelpers<MPInt, int64_t>::from(
                dst,
                &tmp,
                ensure_exact);
        }

        *dst = mpz_get_d(src);
        return ConversionResult::Success;
    }
};

template <typename F>
struct ConversionHelper<F, MPInt, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = const F*;
    using to_ptr = mpz_ptr;

    static constexpr bool is_always_exact = false;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        if (ensure_exact) {
            int64_t tmp = static_cast<int64_t>(*src);
            if (static_cast<F>(tmp) != *src) {
                return ConversionResult::Inexact;
            }
            mpz_set_si(dst, tmp);
        } else { mpz_set_d(dst, *src); }
        return ConversionResult::Success;
    }
};


template <>
struct ConversionHelper<MPInt, MPInt, void>
{
    using from_ptr = mpz_srcptr;
    using to_ptr = mpz_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpz_set(dst, src);
        return ConversionResult::Success;
    }
};


template <typename I>
struct ConversionHelper<MPRational, I, enable_if_t<is_integral_v<I>>>
{
    using from_ptr = mpq_srcptr;
    using to_ptr = I*;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if (mpz_cmp_ui(mpq_denref(src), 1UL)) {
            return ConversionResult::Failed;
        }

        return ConversionHelper<MPInt, I>::convert(dst, mpq_numref(src), ensure_exact);
    }

};


template <typename I>
struct ConversionHelper<I, MPRational, enable_if_t<is_integral_v<I>>>
{
    using from_ptr = const I*;
    using to_ptr = mpq_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        mpq_set_si(dst, *src, 1);
        return ConversionResult::Success;
    }
};


template <typename F>
struct ConversionHelper<MPRational, F, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = mpq_srcptr;
    using to_ptr = F*;

    static constexpr bool is_always_exact = false;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        *dst = mpq_get_d(src);
        return ConversionResult::Success;
    }
};

template <typename F>
struct ConversionHelper<F, MPRational, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = const F*;
    using to_ptr = mpq_ptr;

    static constexpr bool is_always_exact = false;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        if (ensure_exact) {
            // Cannot ensure exact conversion for floating point numbers
            return ConversionResult::Inexact;
        }
        mpq_set_d(dst, *src);
        return ConversionResult::Success;
    }
};

template <>
struct ConversionHelper<MPRational, MPInt, void>
{
    using from_ptr = mpq_srcptr;
    using to_ptr = mpz_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        if (mpz_cmp_ui(mpq_denref(src), 1UL)) {
            return ConversionResult::Failed;
        }

        mpz_set(dst, mpq_numref(src));
        return ConversionResult::Success;
    }

};

template <>
struct ConversionHelper<MPInt, MPRational, void>
{
    using from_ptr = mpz_srcptr;
    using to_ptr = mpq_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpq_set_z(dst, src);
        return ConversionResult::Success;
    }
};


template <>
struct ConversionHelper<MPRational, MPRational, void>
{
    using from_ptr = mpq_srcptr;
    using to_ptr = mpq_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult ConversionResult(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpq_set(dst, src);
        return ConversionResult::Success;
    }
};




template <typename I>
struct ConversionHelper<MPFloat, I, enable_if_t<is_integral_v<I>>>
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = I*;

    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (is_signed_v<I>) {
            *dst = mpfr_get_si(src, MPFR_RNDN);
        } else {
            *dst = mpfr_get_ui(src, MPFR_RNDN);
        }

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~MPFR_FLAGS_INEXACT) { return ConversionResult::Failed; }
        if (ensure_exact && t & MPFR_FLAGS_INEXACT) {
            return ConversionResult::Inexact;
        }

        return ConversionResult::Success;
    }
};

template <typename I>
struct ConversionHelper<I, MPFloat, enable_if_t<is_integral_v<I>>>
{
    using from_ptr = const I*;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if (ensure_exact) {
            if (mpfr_fits_si_p(dst, *src, MPFR_RNDN) != 0) {
                return ConversionResult::Inexact;
            }
        }
        if constexpr (is_signed_v<I>) {
            if constexpr (sizeof(I) < 2) {
               if (ensure_exact && !mpfr_fits_sshort_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            } else if constexpr (sizeof(I) == 4) {
               if (ensure_exact && !mpfr_fits_sint_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            } else if constexpr (sizeof(I) == 8) {
               if (ensure_exact && !mpfr_fits_slong_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            }

            mpfr_set_si(dst, *src, MPFR_RNDN);
            return ConversionResult::Success;
        } else {
            if constexpr (sizeof(I) < 2) {
               if (ensure_exact && !mpfr_fits_ushort_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            } else if constexpr (sizeof(I) == 4) {
               if (ensure_exact && !mpfr_fits_uint_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            } else if constexpr (sizeof(I) == 8) {
               if (ensure_exact && !mpfr_fits_ulong_p(src, MPFR_RNDN)) {
                   return ConversionResult::Inexact;
               }
            }

            mpfr_set_ui(dst, *src, MPFR_RNDN);
            return ConversionResult::Success;
        }

    }
};




template <typename F, bool Nested>
struct ConversionHelpers<MPFloat, F, Nested, enable_if_t<is_floating_point_v<F> > >
{
    static constexpr bool is_nested = Nested;
    static constexpr bool from_exact_convertible() noexcept { return true; }
    static constexpr bool to_exact_convertible() noexcept { return false; }

    static ConversionResult from(mpfr_ptr dst_ptr,
                                 const F* src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);

        mpfr_set_d(dst_ptr, *src_ptr, MPFR_RNDN);
        return ConversionResult::Success;
    }

    static ConversionResult to(F* dst_ptr,
                               mpfr_ptr src_ptr,
                               bool ensure_exact) noexcept
    {
        if constexpr (is_same_v<F, float>) {
            *dst_ptr = mpfr_get_flt(src_ptr, MPFR_RNDN);
        } else { *dst_ptr = mpfr_get_d(src_ptr, MPFR_RNDN); }

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, const F* rhs) noexcept
    {
        return mpfr_cmp_d(lhs, *rhs) == 0;
    }
};


template <bool Nested>
struct ConversionHelpers<MPFloat, MPInt, Nested, void>
{


    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return false; }
    static constexpr bool to_exact_convertible() noexcept { return false; }


    static ConversionResult from(mpfr_ptr dst_ptr,
                                 mpz_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpfr_set_z(dst_ptr, src_ptr, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static ConversionResult to(mpz_ptr dst_ptr,
                               mpfr_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        mpfr_get_z(dst_ptr, src_ptr, MPFR_RNDN);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, mpz_srcptr rhs) noexcept
    {
        return mpfr_cmp_z(lhs, rhs) == 0;
    }
};

template <bool Nested>
struct ConversionHelpers<MPFloat, MPRational, Nested, void>
{
    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return false; }
    static constexpr bool to_exact_convertible() noexcept { return true; }


    static ConversionResult from(mpfr_ptr dst_ptr,
                                 mpq_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        mpfr_set_q(dst_ptr, src_ptr, MPFR_RNDN);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static ConversionResult to(mpq_ptr dst_ptr,
                               mpfr_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        ignore_unused(ensure_exact);
        mpfr_get_q(dst_ptr, src_ptr);
        return ConversionResult::Success;
    }

    static bool compare_equal(mpfr_srcptr lhs, mpq_srcptr rhs) noexcept
    {
        return mpfr_cmp_q(lhs, rhs) == 0;
    }
};

template <bool Nested>
struct ConversionHelpers<MPFloat, MPFloat, Nested, void>
{
    static constexpr bool is_nested = Nested;

    static constexpr bool from_exact_convertible() noexcept { return true; }
    static constexpr bool to_exact_convertible() noexcept { return true; }

    static ConversionResult from(mpfr_ptr dst_ptr,
                                 mpfr_srcptr src_ptr,
                                 bool ensure_exact) noexcept
    {
        mpfr_set(dst_ptr, src_ptr, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }

    static ConversionResult to(mpfr_ptr dst_ptr,
                               mpfr_srcptr src_ptr,
                               bool ensure_exact) noexcept
    {
        return from(dst_ptr, src_ptr, ensure_exact);
    }

    static bool compare_equal(mpfr_srcptr lhs, mpfr_srcptr rhs) noexcept
    {
        return mpfr_cmp(lhs, rhs) == 0;
    }
};



}

#endif //ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H