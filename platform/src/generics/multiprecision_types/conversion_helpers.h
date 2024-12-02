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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult
    convert(to_ptr dst, from_ptr src, bool ensure_exact)
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
                                    from_ptr src,
                                    bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        if constexpr (is_signed_v<I>) { mpz_set_si(dst, *src); } else {
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst,
                                    from_ptr src,
                                    bool ensure_exact)
    {
        if (ensure_exact) {
            int64_t tmp;
            auto first_result = ConversionHelper<int64_t, F>::from(
                &tmp,
                src,
                ensure_exact);
            if (first_result != ConversionResult::Success) {
                return first_result;
            }
            return ConversionHelper<int64_t, MPInt>::convert(
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst,
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpz_set(dst, src);
        return ConversionResult::Success;
    }
};


template <typename I>
struct ConversionHelper<MPRational, I, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = mpq_srcptr;
    using to_ptr = I*;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        if (mpz_cmp_ui(mpq_denref(src), 1UL)) {
            return ConversionResult::Failed;
        }

        return ConversionHelper<MPInt, I>::convert(
            dst,
            mpq_numref(src),
            ensure_exact);
    }

};


template <typename I>
struct ConversionHelper<I, MPRational, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = const I*;
    using to_ptr = mpq_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst,
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst,
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
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

    static constexpr bool is_possible = true;
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

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst,
                                             from_ptr src,
                                             bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpq_set(dst, src);
        return ConversionResult::Success;
    }
};


template <typename I>
struct ConversionHelper<MPFloat, I, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = I*;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (is_signed_v<I>) {
            *dst = mpfr_get_si(src, MPFR_RNDN);
        } else { *dst = mpfr_get_ui(src, MPFR_RNDN); }

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
struct ConversionHelper<I, MPFloat, enable_if_t<is_integral_v<I> > >
{
    using from_ptr = const I*;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if constexpr (is_signed_v<I>) {
            if constexpr (sizeof(I) < 2) {
                if (ensure_exact && !mpfr_fits_sshort_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 4) {
                if (ensure_exact && !mpfr_fits_sint_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 8) {
                if (ensure_exact && !mpfr_fits_slong_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            }

            mpfr_set_si(dst, *src, MPFR_RNDN);
            return ConversionResult::Success;
        } else {
            if constexpr (sizeof(I) < 2) {
                if (ensure_exact && !mpfr_fits_ushort_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 4) {
                if (ensure_exact && !mpfr_fits_uint_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            } else if constexpr (sizeof(I) == 8) {
                if (ensure_exact && !mpfr_fits_ulong_p(dst, MPFR_RNDN)) {
                    return ConversionResult::Inexact;
                }
            }

            mpfr_set_ui(dst, *src, MPFR_RNDN);
            return ConversionResult::Success;
        }

    }
};

template <typename F>
struct ConversionHelper<MPFloat, F, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = const F*;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        *dst = mpfr_get_d(src, MPFR_RNDN);
        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }
};


template <typename F>
struct ConversionHelper<F, MPFloat, enable_if_t<is_floating_point_v<F> > >
{
    using from_ptr = const F*;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        if (ensure_exact) {
            if (mpfr_fits_d_p(dst, *src, MPFR_RNDN) != 0) {
                return ConversionResult::Inexact;
            }
        }
        mpfr_set_d(dst, *src, MPFR_RNDN);
        return ConversionResult::Success;
    }
};

template <>
struct ConversionHelper<MPFloat, MPInt, void>
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = mpz_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        mpfr_get_z(dst, src, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }
};

template <>
struct ConversionHelper<MPInt, MPFloat, void>
{
    using from_ptr = mpz_srcptr;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        mpfr_set_z(dst, src, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }
};


template <>
struct ConversionHelper<MPFloat, MPRational, void>
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = mpq_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = true;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        ignore_unused(ensure_exact);
        mpfr_get_q(dst, src);
        return ConversionResult::Success;
    }
};

template <>
struct ConversionHelper<MPRational, MPFloat, void>
{
    using from_ptr = mpq_srcptr;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        mpfr_set_q(dst, src, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }
};


template <>
struct ConversionHelper<MPFloat, MPFloat, void>
{
    using from_ptr = mpfr_srcptr;
    using to_ptr = mpfr_ptr;

    static constexpr bool is_possible = true;
    static constexpr bool is_always_exact = false;

    static ConversionResult convert(to_ptr dst, from_ptr src, bool ensure_exact)
    {
        mpfr_set(dst, src, MPFR_RNDN);

        constexpr mpfr_flags_t flags = (MPFR_FLAGS_INEXACT | MPFR_FLAGS_OVERFLOW
            | MPFR_FLAGS_UNDERFLOW);

        auto t = mpfr_flags_test(MPFR_FLAGS_ALL);
        mpfr_flags_clear(MPFR_FLAGS_ALL);
        if (t & ~flags) { return ConversionResult::Failed; }
        if (ensure_exact && t & flags) { return ConversionResult::Inexact; }

        return ConversionResult::Success;
    }
};

}

#endif //ROUGHPY_GENERICS_INTERNAL_MULTIPRECISION_CONVERSION_HELPERS_H