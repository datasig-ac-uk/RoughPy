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

//
// Created by user on 02/08/23.
//

#ifndef ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_H_
#define ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_H_

#include "scalar_blas_defs.h"
#include "blas.h"

#define RPY_LPK_FUNC_(NAME) RPY_JOIN(LAPACKE_, NAME)
#define RPY_LPK_FUNC(NAME) \
    RPY_LPK_FUNC_(RPY_JOIN(RPY_LPK_SPRX, RPY_JOIN(NAME, _work)))

#define RPY_LPK_JOB_ARG(ARG) *ARG
#define RPY_LPK_TPS_ARG(ARG) ARG
#define RPY_LPK_UPL_ARG(ARG) ARG
#define RPY_LPK_INT_ARG(ARG) ARG
#define RPY_LPK_SCA_ARG(ARG) convert_scalar(ARG)
#define RPY_LPK_RSC_ARG(ARG) ARG
#define RPY_LPK_PTR_ARG(ARG) reinterpret_cast<lpk_scalar*>(ARG)
#define RPY_LPK_RSP_ARG(ARG) ARG
#define RPY_LPK_CPT_ARG(ARG) reinterpret_cast<const lpk_scalar*>(ARG)
#define RPY_LPK_IPR_ARG(ARG) ARG
#define RPY_LPK_IRF_ARG(ARG) &ARG
#define RPY_LPK_CIP_ARG(ARG) ARG
#define RPY_LPK_WORK m_work.data()
#define RPY_LPK_IWRK m_iwork.data()
#define RPY_LPK_RWRK m_rwork.data()

#define RPY_LPK_CALL_(ROUTINE, INFO, LAYOUT, ...) \
    INFO = ROUTINE(LAYOUT, __VA_ARGS__)
#define RPY_LPK_CALL(ROUTINE, INFO, LAYOUT, ...) \
    INFO = RPY_LPK_FUNC(ROUTINE)(LAYOUT, __VA_ARGS__)

namespace rpy {
namespace scalars {
namespace lapack {

using ::rpy::blas::BlasLayout;
using ::rpy::blas::BlasTranspose;
using ::rpy::blas::BlasUpLo;
using ::rpy::blas::complex32;
using ::rpy::blas::complex64;
using ::rpy::blas::integer;
using ::rpy::blas::logical;





template <typename S, typename R>
struct lapack_func_workspace;

template <>
struct lapack_func_workspace<float, float> {
    std::vector<float> m_work;
    std::vector<integer> m_iwork;
    integer lwork;

    using lpk_scalar = float;

    void reset_workspace()
    {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
    }

    void resize_workspace(bool iwork = false)
    {
        lwork = static_cast<integer>(m_work[0]);
        m_work.resize(lwork);
        if (iwork) { m_iwork.resize(m_iwork[0]); }
    }
};

template <>
struct lapack_func_workspace<double, double> {
    std::vector<double> m_work;
    std::vector<integer> m_iwork;
    integer lwork;

    using lpk_scalar = double;

    void reset_workspace()
    {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
    }
    void resize_workspace(bool iwork = false)
    {
        lwork = static_cast<integer>(m_work[0]);
        m_work.resize(lwork);
        if (iwork) { m_iwork.resize(m_iwork[0]); }
    }
};

template <>
struct lapack_func_workspace<scalars::float_complex, float> {
    std::vector<complex32> m_work;
    std::vector<float> m_rwork;
    std::vector<integer> m_iwork;
    integer lwork;

    using lpk_scalar = complex32;

    void reset_workspace()
    {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
        m_rwork.resize(1);
    }

    void resize_workspace(bool rwork = false, bool iwork = false)
    {
        lwork = static_cast<integer>(m_work[0].real);
        m_work.resize(lwork);
        if (iwork) { m_iwork.resize(m_iwork[0]); }
        if (rwork) { m_rwork.resize(static_cast<integer>(m_rwork[0])); }
    }
};

template <>
struct lapack_func_workspace<scalars::double_complex, double> {
    std::vector<complex64> m_work;
    std::vector<double> m_rwork;
    std::vector<integer> m_iwork;
    integer lwork;

    using lpk_scalar = complex64;

    void reset_workspace()
    {
        lwork = -1;
        m_work.resize(1);
        m_iwork.resize(1);
        m_rwork.resize(1);
    }

    void resize_workspace(bool rwork = false, bool iwork = false)
    {
        lwork = static_cast<integer>(m_work[0].real);
        m_work.resize(lwork);
        if (iwork) { m_iwork.resize(m_iwork[0]); }
        if (rwork) { m_rwork.resize(static_cast<integer>(m_rwork[0])); }
    }
};

template <typename S, typename R>
struct lapack_funcs : lapack_func_workspace<S, R> {
    using scalar = S;
    using real_scalar = R;

    using typename lapack_func_workspace<S, R>::lpk_scalar;

    static inline void
    handle_illegal_parameter_error(const char* method, integer arg)
    {
        std::stringstream ss;
        ss << "invalid argument " << arg << " in call to " << method;
        RPY_THROW(std::invalid_argument, ss.str());
    }

    void
    gesv(BlasLayout layout, const integer n, const integer nrhs, scalar* A,
         const integer lda, integer* ipiv, scalar* B, const integer ldb);

    void
    syev(BlasLayout layout, const char* jobz, blas::BlasUpLo uplo,
         const integer n, scalar* RPY_RESTRICT A, const integer lda,
         real_scalar* RPY_RESTRICT w);

    void
    geev(BlasLayout layout, const char* joblv, const char* jobvr,
         const integer n, scalar* RPY_RESTRICT A, const integer lda,
         scalar* RPY_RESTRICT wr, scalar* RPY_RESTRICT RPY_UNUSED_VAR wi,
         scalar* RPY_RESTRICT vl, const integer ldvl, scalar* RPY_RESTRICT vr,
         const integer ldvr);

    void
    gesvd(BlasLayout layout, const char* jobu, const char* jobvt,
          const integer m, const integer n, scalar* RPY_RESTRICT A,
          const integer lda, real_scalar* RPY_RESTRICT s,
          scalar* RPY_RESTRICT u, const integer ldu, scalar* RPY_RESTRICT vt,
          const integer ldvt);

    void
    gesdd(BlasLayout layout, const char* jobz, const integer m, const integer n,
          scalar* RPY_RESTRICT A, const integer lda,
          real_scalar* RPY_RESTRICT s, scalar* RPY_RESTRICT u,
          const integer ldu, scalar* RPY_RESTRICT vt, const integer ldvt);

    void
    gels(BlasLayout layout, BlasTranspose trans, const integer m,
         const integer n, const integer nrhs, scalar* RPY_RESTRICT A,
         const integer lda, scalar* RPY_RESTRICT B, const integer ldb);

    void
    gelsy(BlasLayout layout, const integer m, const integer n,
          const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
          scalar* RPY_RESTRICT B, const integer ldb, integer* RPY_RESTRICT jpvt,
          const real_scalar& rcond, integer& rank);

    void
    gelss(BlasLayout layout, const integer m, const integer n,
          const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
          scalar* RPY_RESTRICT B, const integer ldb,
          real_scalar* RPY_RESTRICT s, const real_scalar& rcond, integer& rank);

    void
    gelsd(BlasLayout layout, const integer m, const integer n,
          const integer nrhs, scalar* RPY_RESTRICT A, const integer lda,
          scalar* RPY_RESTRICT B, const integer ldb,
          real_scalar* RPY_RESTRICT s, const real_scalar& rcond, integer& rank);
};

}// namespace lapack
}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_H_
