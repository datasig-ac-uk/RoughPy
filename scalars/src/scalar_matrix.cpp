// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
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

rpy::scalars::ScalarMatrix::ScalarMatrix(
        const rpy::scalars::ScalarType* type, rpy::deg_t rows, rpy::deg_t cols,
        rpy::scalars::MatrixLayout layout
)
    : ScalarArray(type, (void*) nullptr, 0), m_layout(layout), m_nrows(rows),
      m_ncols(cols)
{
    if (p_type != nullptr && m_nrows > 0 && m_ncols > 0) {
        const auto size = m_nrows * m_ncols;
        ScalarPointer::operator=(p_type->allocate(size));
        m_size = size;
    }
}
rpy::scalars::ScalarMatrix::ScalarMatrix(
        rpy::deg_t rows, rpy::deg_t cols, rpy::scalars::ScalarArray&& array,
        rpy::scalars::MatrixLayout layout
)
    : ScalarArray(std::move(array)), m_layout(layout), m_nrows(rows),
      m_ncols(cols)
{
    RPY_CHECK(m_nrows >= 0 && m_ncols >= 0);
    RPY_CHECK(
            static_cast<dimn_t>(m_nrows) * static_cast<dimn_t>(m_ncols)
            == m_size
    );
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

static void transpose_fallback(rpy::scalars::ScalarMatrix& matrix)
{
    const auto M = matrix.nrows();
    const auto N = matrix.ncols();

    const auto* type = matrix.type();
    rpy::scalars::ScalarPointer ptr(matrix);
    if (M == N) {
        for (rpy::deg_t i = 0; i < M; ++i) {
            for (rpy::deg_t j = i + 1; j < N; ++j) {
                type->swap(
                        ptr + static_cast<rpy::dimn_t>(i * N + j),
                        ptr + static_cast<rpy::dimn_t>(j * N + i), 1
                );
            }
        }
    } else {
        rpy::scalars::ScalarMatrix tmp(type, N, M, matrix.layout());

        for (rpy::deg_t i = 0; i < M; ++i) {
            for (rpy::deg_t j = 0; j < N; ++j) {
                type->convert_copy(
                        tmp + static_cast<rpy::dimn_t>(j * M + i),
                        ptr + static_cast<rpy::dimn_t>(i * N + j), 1
                );
            }
        }

        matrix = std::move(tmp);
    }
}

rpy::scalars::ScalarMatrix::~ScalarMatrix()
{
    if (is_owning()) { p_type->free(*this, m_size); }
}

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::ScalarMatrix

#include <roughpy/platform/serialization_instantiations.inl>
