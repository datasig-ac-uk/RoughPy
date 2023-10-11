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

#ifndef ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_COMPLEX_DOUBLE_CPP_
#define ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_COMPLEX_DOUBLE_CPP_

#include "lapack.h"


#ifndef ROUGHPY_DISABLE_BLAS

#define RPY_LPK_SPRX z


static constexpr rpy::blas::complex64
convert_scalar(const rpy::scalars::double_complex& arg)
{
    return {arg.real(), arg.imag()};
}

namespace rpy {
namespace scalars {
namespace lapack {


template <>
void lapack_funcs<double_complex, double>::gesv(
        BlasLayout layout, const integer n, const integer nrhs,
        double_complex* A, const integer lda, integer* ipiv,
        double_complex* B, const integer ldb
)
{
    integer info = 0;
    RPY_LPK_CALL(
            gesv, info, layout, RPY_LPK_INT_ARG(n), RPY_LPK_INT_ARG(nrhs),
            RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda), RPY_LPK_IPR_ARG(ipiv),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb)
    );
    if (info < 0) {
        handle_illegal_parameter_error("gesv", -info);
    } else if (info > 0) {
        std::stringstream ss;
        ss << "component" << info
           << " on the diagonal of U is zero so the matrix is singular";
        RPY_THROW(std::runtime_error, ss.str());
    }
}
template <>
void lapack_funcs<double_complex, double>::syev(
        BlasLayout layout, const char* jobz, blas::BlasUpLo uplo,
        const integer n, double_complex* A, const integer lda, double* w
)
{
    integer info = 0;
    reset_workspace();
    RPY_LPK_CALL(
            heev, info, layout, RPY_LPK_JOB_ARG(jobz), RPY_LPK_UPL_ARG(uplo),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_RSP_ARG(w), RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork),
            RPY_LPK_RWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LPK_CALL(
            heev, info, layout, RPY_LPK_JOB_ARG(jobz), RPY_LPK_UPL_ARG(uplo),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_RSP_ARG(w), RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork),
            RPY_LPK_RWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("syev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<double_complex, double>::geev(
        BlasLayout layout, const char* jobvl, const char* jobvr,
        const integer n, double_complex* A, const integer lda,
        double_complex* wr, double_complex* RPY_UNUSED_VAR wi,
        double_complex* vl, const integer ldvl,
        double_complex* vr, const integer ldvr
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            geev, info, layout, RPY_LPK_JOB_ARG(jobvl), RPY_LPK_JOB_ARG(jobvr),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(wr), RPY_LPK_PTR_ARG(vl), RPY_LPK_INT_ARG(ldvl),
            RPY_LPK_PTR_ARG(vr), RPY_LPK_INT_ARG(ldvr), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LPK_CALL(
            geev, info, layout, RPY_LPK_JOB_ARG(jobvl), RPY_LPK_JOB_ARG(jobvr),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(wr), RPY_LPK_PTR_ARG(vl), RPY_LPK_INT_ARG(ldvl),
            RPY_LPK_PTR_ARG(vr), RPY_LPK_INT_ARG(ldvr), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("geev", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "the eigenvalues failed to converge");
    }
}
template <>
void lapack_funcs<double_complex, double>::gesvd(
        BlasLayout layout, const char* jobu, const char* jobvt, const integer m,
        const integer n, double_complex* A, const integer lda, double* s,
        double_complex* u, const integer ldu,
        double_complex* vt, const integer ldvt
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gesvd, info, layout, RPY_LPK_JOB_ARG(jobu), RPY_LPK_JOB_ARG(jobvt),
            RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A),
            RPY_LPK_INT_ARG(lda), RPY_LPK_RSP_ARG(s), RPY_LPK_PTR_ARG(u),
            RPY_LPK_INT_ARG(ldu), RPY_LPK_PTR_ARG(vt), RPY_LPK_INT_ARG(ldvt),
            RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LPK_CALL(
            gesvd, info, layout, RPY_LPK_JOB_ARG(jobu), RPY_LPK_JOB_ARG(jobvt),
            RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A),
            RPY_LPK_INT_ARG(lda), RPY_LPK_RSP_ARG(s), RPY_LPK_PTR_ARG(u),
            RPY_LPK_INT_ARG(ldu), RPY_LPK_PTR_ARG(vt), RPY_LPK_INT_ARG(ldvt),
            RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("gesvd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<double_complex, double>::gesdd(
        BlasLayout layout, const char* jobz, const integer m, const integer n,
        double_complex* A, const integer lda, double* s,
        double_complex* u, const integer ldu,
        double_complex* vt, const integer ldvt
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gesdd, info, layout, RPY_LPK_JOB_ARG(jobz), RPY_LPK_INT_ARG(m),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_RSP_ARG(s), RPY_LPK_PTR_ARG(u), RPY_LPK_INT_ARG(ldu),
            RPY_LPK_PTR_ARG(vt), RPY_LPK_INT_ARG(ldvt), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK, RPY_LPK_IWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true, true);

    RPY_LPK_CALL(
            gesdd, info, layout, RPY_LPK_JOB_ARG(jobz), RPY_LPK_INT_ARG(m),
            RPY_LPK_INT_ARG(n), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_RSP_ARG(s), RPY_LPK_PTR_ARG(u), RPY_LPK_INT_ARG(ldu),
            RPY_LPK_PTR_ARG(vt), RPY_LPK_INT_ARG(ldvt), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK, RPY_LPK_IWRK
    );
    if (info == -4) {
        RPY_THROW(std::invalid_argument, "matrix A contains a NaN value");
    } else if (info < 0) {
        handle_illegal_parameter_error("gesdd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "singular values failed to converge");
    }
}
template <>
void lapack_funcs<double_complex, double>::gels(
        BlasLayout layout, blas::BlasTranspose trans, const integer m,
        const integer n, const integer nrhs, double_complex* A,
        const integer lda, double_complex* B, const integer ldb
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gels, info, layout, RPY_LPK_TPS_ARG(trans), RPY_LPK_INT_ARG(m),
            RPY_LPK_INT_ARG(n), RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A),
            RPY_LPK_INT_ARG(lda), RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb),
            RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork)
    );
    RPY_CHECK(info == 0);
    resize_workspace();

    RPY_LPK_CALL(
            gels, info, layout, RPY_LPK_TPS_ARG(trans), RPY_LPK_INT_ARG(m),
            RPY_LPK_INT_ARG(n), RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A),
            RPY_LPK_INT_ARG(lda), RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb),
            RPY_LPK_WORK, RPY_LPK_INT_ARG(lwork)
    );
    if (info < 0) {
        handle_illegal_parameter_error("gels", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<double_complex, double>::gelsy(
        BlasLayout layout, const integer m, const integer n, const integer nrhs,
        double_complex* A, const integer lda, double_complex* B,
        const integer ldb, integer* jpvt, const double& rcond, integer& rank
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gelsy, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_IPR_ARG(jpvt),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LPK_CALL(
            gelsy, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_IPR_ARG(jpvt),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("gelsy", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "matrix does not have full rank");
    }
}
template <>
void lapack_funcs<double_complex, double>::gelss(
        BlasLayout layout, const integer m, const integer n, const integer nrhs,
        double_complex* A, const integer lda, double_complex* B,
        const integer ldb, double* s, const double& rcond, integer& rank
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gelss, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_RSP_ARG(s),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true);

    RPY_LPK_CALL(
            gelss, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_RSP_ARG(s),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("gelss", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}
template <>
void lapack_funcs<double_complex, double>::gelsd(
        BlasLayout layout, const integer m, const integer n, const integer nrhs,
        double_complex* A, const integer lda, double_complex* B,
        const integer ldb, double* s, const double& rcond, integer& rank
)
{
    integer info = 0;

    reset_workspace();
    RPY_LPK_CALL(
            gelsd, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_RSP_ARG(s),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK, RPY_LPK_IWRK
    );
    RPY_CHECK(info == 0);
    resize_workspace(true, true);

    RPY_LPK_CALL(
            gelsd, info, layout, RPY_LPK_INT_ARG(m), RPY_LPK_INT_ARG(n),
            RPY_LPK_INT_ARG(nrhs), RPY_LPK_PTR_ARG(A), RPY_LPK_INT_ARG(lda),
            RPY_LPK_PTR_ARG(B), RPY_LPK_INT_ARG(ldb), RPY_LPK_RSP_ARG(s),
            RPY_LPK_RSC_ARG(rcond), RPY_LPK_IRF_ARG(rank), RPY_LPK_WORK,
            RPY_LPK_INT_ARG(lwork), RPY_LPK_RWRK, RPY_LPK_IWRK
    );
    if (info < 0) {
        handle_illegal_parameter_error("gelsd", -info);
    } else if (info > 0) {
        RPY_THROW(std::runtime_error, "algorithm for computing svd failed");
    }
}

}// namespace lapack
}// namespace scalars
}// namespace rpy

#endif
#endif// ROUGHPY_SCALARS_SRC_LINEAR_ALGEBRA_LAPACK_COMPLEX_DOUBLE_CPP_
