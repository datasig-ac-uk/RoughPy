// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_SCALARS_TYPES_H_
#define ROUGHPY_SCALARS_TYPES_H_

#include <complex>

#include <Eigen/Core>
#include <libalgebra_lite/coefficients.h>
#include <libalgebra_lite/packed_integer.h>
#include <libalgebra_lite/polynomial.h>
#include <libalgebra_lite/polynomial_basis.h>


#include "scalar_pointer.h"
#include "scalar_type.h"
#include "scalar.h"

#ifdef LAL_NO_USE_GMP
#  define RPY_USING_GMP 0
#else
#  define RPY_USING_GMP 1
#endif

namespace rpy {
namespace scalars {

/// IEEE half-precision floating point type
using Eigen::half;

/// BFloat16 (truncated) floating point type
using Eigen::bfloat16;

/// Rational scalar type
using rational_scalar_type = lal::rational_field::scalar_type;

/// half-precision complex float
using half_complex = std::complex<half>;

/// BFloat16 complex float - probably not supported anywhere
using bf16_complex = std::complex<bfloat16>;

/// Single precision complex float
using float_complex = std::complex<float>;

/// double precision complex float
using double_complex = std::complex<double>;

/// Monomial key-type of polynomials
using monomial = lal::monomial;

/// Indeterminate type for monomials
using indeterminate_type = typename monomial::letter_type;

/// Polynomial (with rational coefficients) scalar type
using rational_poly_scalar = lal::polynomial<lal::rational_field>;

namespace dtl {
#define ROUGHPY_MAKE_TYPE_ID_OF(TYPE, NAME)                                    \
    template <>                                                                \
    struct ROUGHPY_SCALARS_EXPORT type_id_of_impl<TYPE> {                                  \
        static const string& get_id() noexcept;                                \
    }



ROUGHPY_MAKE_TYPE_ID_OF(half, "f16");

ROUGHPY_MAKE_TYPE_ID_OF(bfloat16, "bf16");

ROUGHPY_MAKE_TYPE_ID_OF(rational_scalar_type, "Rational");

ROUGHPY_MAKE_TYPE_ID_OF(rational_poly_scalar, "RationalPoly");

#undef ROUGHPY_MAKE_TYPE_ID_OF




template <>
struct ROUGHPY_SCALARS_EXPORT scalar_type_holder<rational_scalar_type> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct ROUGHPY_SCALARS_EXPORT scalar_type_holder<rational_poly_scalar> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct ROUGHPY_SCALARS_EXPORT scalar_type_holder<half> {
    static const ScalarType* get_type() noexcept;
};

template <>
struct ROUGHPY_SCALARS_EXPORT scalar_type_holder<bfloat16> {
    static const ScalarType* get_type() noexcept;
};

}// namespace dtl

namespace dtl {

template <typename T>
struct type_id_of_impl {
    static const string& get_id()
    {
        /*
         * The fallback implementation gets the type id by looking for the
         * type and querying for the id. This should not fail unless no type
         * is associated with T, since all the cases where ScalarType::of<T>
         * () is allowed to be nullptr are integer types, which are
         * overloaded below.
         */
        const auto* type = ScalarType::of<T>();
        RPY_CHECK(type != nullptr);
        return type->id();
    }
};

}// namespace dtl


namespace dtl {

template <typename T>
struct type_of_T_defined {
    static T cast(ScalarPointer scalar)
    {
        const auto* tp = ScalarType::of<T>();
        if (tp == scalar.type()) { return *scalar.raw_cast<const T*>(); }
        if (tp == scalar.type()->rational_type()) {
            return *scalar.raw_cast<const T*>();
        }

        T result;
        ScalarPointer dst(tp, &result);
        tp->convert_copy(dst, scalar, 1);
        return result;
    }
};

template <typename T>
struct type_of_T_not_defined {
    static T cast(ScalarPointer scalar)
    {
        T result;
        scalar.type()->convert_copy({nullptr, &result}, scalar, 1,
                                    type_id_of<T>());
        return result;
    }
};

}// namespace dtl

template <typename T>
inline remove_cv_ref_t<T> scalar_cast(const Scalar& scalar)
{
    if (scalar.is_zero()) { return T(0); }
    using bare_t = remove_cv_ref_t<T>;
    using impl_t = detected_or_t<dtl::type_of_T_not_defined<bare_t>,
                                 dtl::type_of_T_defined, bare_t>;

    // Now we are sure that scalar.type() != nullptr
    // and scalar.ptr() != nullptr
    return impl_t::cast(scalar.to_pointer());
}
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_TYPES_H_
