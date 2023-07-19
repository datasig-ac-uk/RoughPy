// Copyright (c) 2023 RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
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
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//
// Created by user on 12/04/23.
//

#include <roughpy/scalars/scalar_matrix.h>

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_blas.h>
#include <roughpy/scalars/scalar_type.h>

rpy::scalars::ScalarMatrix::ScalarMatrix() : ScalarArray() {}

rpy::scalars::ScalarMatrix::ScalarMatrix(const rpy::scalars::ScalarType* type,
                                         rpy::deg_t rows, rpy::deg_t cols,
                                         rpy::scalars::MatrixStorage storage,
                                         rpy::scalars::MatrixLayout layout)
    : ScalarArray(type, (void*) nullptr, 0), m_storage(storage),
      m_layout(layout), m_nrows(rows), m_ncols(cols)
{
    if (p_type != nullptr && m_nrows > 0 && m_ncols > 0) {
        const auto size = m_nrows * m_ncols;
        ScalarPointer::operator=(p_type->allocate(size));
        m_size = size;
    }
}
rpy::scalars::ScalarMatrix::ScalarMatrix(rpy::deg_t rows, rpy::deg_t cols,
                                         rpy::scalars::ScalarArray&& array,
                                         rpy::scalars::MatrixStorage storage,
                                         rpy::scalars::MatrixLayout layout)
    : ScalarArray(std::move(array)), m_storage(storage), m_layout(layout),
      m_nrows(rows), m_ncols(cols)
{
    RPY_CHECK(m_nrows >= 0 && m_ncols >= 0);
    RPY_CHECK(static_cast<dimn_t>(m_nrows) * static_cast<dimn_t>(m_ncols)
              == m_size);
}
rpy::scalars::ScalarMatrix rpy::scalars::ScalarMatrix::row(rpy::deg_t i)
{
    return rpy::scalars::ScalarMatrix(nullptr, 0, 0);
}
rpy::scalars::ScalarMatrix rpy::scalars::ScalarMatrix::row(rpy::deg_t i) const
{
    return rpy::scalars::ScalarMatrix(nullptr, 0, 0);
}
rpy::scalars::ScalarMatrix rpy::scalars::ScalarMatrix::col(rpy::deg_t i)
{
    return rpy::scalars::ScalarMatrix(nullptr, 0, 0);
}
rpy::scalars::ScalarMatrix rpy::scalars::ScalarMatrix::col(rpy::deg_t i) const
{
    return rpy::scalars::ScalarMatrix(nullptr, 0, 0);
}
rpy::scalars::ScalarPointer rpy::scalars::ScalarMatrix::data() const
{
    return rpy::scalars::ScalarPointer();
}
rpy::scalars::ScalarPointer rpy::scalars::ScalarMatrix::data()
{
    return rpy::scalars::ScalarPointer();
}
rpy::scalars::ScalarMatrix rpy::scalars::ScalarMatrix::to_full() const
{
    return to_full(m_layout);
}

static void transpose_fallback(rpy::scalars::ScalarMatrix& matrix)
{
    const auto M = matrix.nrows();
    const auto N = matrix.ncols();

    const auto* type = matrix.type();
    rpy::scalars::ScalarPointer ptr(matrix);
    if (M == N) {
        for (rpy::deg_t i = 0; i < M; ++i) {
            for (rpy::deg_t j = i + 1; j < N; ++j) {
                type->swap(ptr + static_cast<rpy::dimn_t>(i * N + j),
                           ptr + static_cast<rpy::dimn_t>(j * N + i));
            }
        }
    } else {
        rpy::scalars::ScalarMatrix tmp(type, N, M, matrix.storage(),
                                       matrix.layout());

        for (rpy::deg_t i = 0; i < M; ++i) {
            for (rpy::deg_t j = 0; j < N; ++j) {
                type->convert_copy(tmp + static_cast<rpy::dimn_t>(j * M + i),
                                   ptr + static_cast<rpy::dimn_t>(i * N + j),
                                   1);
            }
        }

        matrix = std::move(tmp);
    }
}

rpy::scalars::ScalarMatrix
rpy::scalars::ScalarMatrix::to_full(rpy::scalars::MatrixLayout layout) const
{
    if (p_type == nullptr) {
        RPY_THROW(std::invalid_argument, "cannot allocate matrix with no type");
    }

    ScalarMatrix result(p_type, m_nrows, m_ncols, m_storage, layout);
    to_full(result);
    return result;
}

void rpy::scalars::ScalarMatrix::to_full(rpy::scalars::ScalarMatrix& into) const
{
    const auto layout = into.layout();

    ScalarPointer iptr(*this);
    ScalarPointer optr(into);
    switch (m_storage) {
        case MatrixStorage::FullMatrix:
            if (m_layout == layout) {
                p_type->convert_copy(into.ptr(), iptr, m_size);
            } else {
                auto blas = p_type->get_blas();
                if (blas) {
                    blas->transpose(into);
                } else {
                    transpose_fallback(into);
                }
            }
            break;
        case MatrixStorage::UpperTriangular: {
            auto mindim = std::min(m_nrows, m_ncols);
            dimn_t offset = 0;
            switch (layout) {
                case MatrixLayout::CStype:
                    for (deg_t i = 0; i < mindim; ++i) {
                        p_type->convert_copy(
                                optr + static_cast<dimn_t>(i * (1 + m_ncols)),
                                iptr + offset, m_ncols - i);
                        offset += m_ncols;
                    }
                    break;
                case MatrixLayout::FStype:
                    for (deg_t i = 1; i <= mindim; ++i) {
                        p_type->convert_copy(
                                optr + static_cast<dimn_t>(i * m_nrows),
                                iptr + offset, i);
                        offset += i;
                    }
                    break;
            }
            break;
        }
        case MatrixStorage::LowerTriangular: {
            auto mindim = std::min(m_nrows, m_ncols);
            dimn_t offset = 0;
            switch (layout) {
                case MatrixLayout::CStype:
                    for (deg_t i = 1; i <= mindim; ++i) {
                        p_type->convert_copy(
                                optr + static_cast<dimn_t>(i * m_ncols),
                                iptr + offset, i);
                        offset += i;
                    }
                    break;
                case MatrixLayout::FStype:
                    for (deg_t i = 0; i < mindim; ++i) {
                        p_type->convert_copy(
                                optr + static_cast<dimn_t>(i * (1 + m_nrows)),
                                iptr + offset, m_ncols - i);
                        offset += m_ncols - i;
                    }
            }
            break;
        }
        case MatrixStorage::Diagonal: {
            auto mindim = std::min(m_nrows, m_ncols);
            auto stride = (layout == MatrixLayout::CStype) ? m_ncols : m_nrows;
            for (deg_t i = 0; i < mindim; ++i) {
                p_type->convert_copy(
                        optr + static_cast<dimn_t>(i * (1 + stride)),
                        iptr + static_cast<dimn_t>(i), 1);
            }
            break;
        }
    }
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::ScalarMatrix

#include <roughpy/platform/serialization_instantiations.inl>
