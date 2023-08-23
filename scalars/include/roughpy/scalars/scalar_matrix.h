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

#ifndef ROUGHPY_SCALARS_SCALAR_MATRIX_H_
#define ROUGHPY_SCALARS_SCALAR_MATRIX_H_

#include "scalar_array.h"
#include "scalars_fwd.h"

#include <roughpy/platform/serialization.h>

namespace rpy {
namespace scalars {

using matrix_dim_t = int;

enum class MatrixLayout
{
    RowMajor,
    ColumnMajor
};

class RPY_EXPORT ScalarMatrix : public scalars::ScalarArray
{
    MatrixLayout m_layout = MatrixLayout::RowMajor;
    matrix_dim_t m_nrows = 0;
    matrix_dim_t m_ncols = 0;

public:
    ScalarMatrix();

    ScalarMatrix(
            const ScalarType* type, matrix_dim_t rows, matrix_dim_t cols,
            MatrixLayout = MatrixLayout::RowMajor
    );

    ScalarMatrix(
            matrix_dim_t rows, matrix_dim_t cols, ScalarArray&& array,
            MatrixLayout layout = MatrixLayout::RowMajor
    );

    ~ScalarMatrix();

    RPY_NO_DISCARD constexpr matrix_dim_t nrows() const noexcept
    {
        return m_nrows;
    }

    RPY_NO_DISCARD constexpr matrix_dim_t ncols() const noexcept
    {
        return m_ncols;
    }

    RPY_NO_DISCARD constexpr MatrixLayout layout() const noexcept
    {
        return m_layout;
    }

    constexpr void layout(MatrixLayout new_layout) noexcept
    {
        m_layout = new_layout;
    }

    RPY_NO_DISCARD constexpr matrix_dim_t leading_dimension() const noexcept
    {
        return (m_layout == MatrixLayout::RowMajor) ? m_ncols : m_nrows;
    }

    RPY_NO_DISCARD ScalarMatrix row(matrix_dim_t i);
    RPY_NO_DISCARD ScalarMatrix row(matrix_dim_t i) const;
    RPY_NO_DISCARD ScalarMatrix col(matrix_dim_t i);
    RPY_NO_DISCARD ScalarMatrix col(matrix_dim_t i) const;

    RPY_NO_DISCARD ScalarPointer data() const;
    RPY_NO_DISCARD ScalarPointer data();

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(ScalarMatrix)
{
    RPY_SERIAL_SERIALIZE_NVP("layout", m_layout);
    RPY_SERIAL_SERIALIZE_NVP("rows", m_nrows);
    RPY_SERIAL_SERIALIZE_NVP("cols", m_ncols);
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
}

}// namespace scalars
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(
        rpy::scalars::ScalarMatrix,
        rpy::serial::specialization::member_serialize
)

#endif// ROUGHPY_SCALARS_SCALAR_MATRIX_H_
