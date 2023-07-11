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

#ifndef ROUGHPY_SCALARS_SCALAR_MATRIX_H_
#define ROUGHPY_SCALARS_SCALAR_MATRIX_H_

#include "scalar_array.h"
#include "scalars_fwd.h"

#include <roughpy/platform/serialization.h>

namespace rpy {
namespace scalars {

enum class MatrixStorage
{
    FullMatrix,
    UpperTriangular,
    LowerTriangular,
    Diagonal
};

enum class MatrixLayout
{
    CStype,
    FStype
};

class RPY_EXPORT ScalarMatrix : public scalars::ScalarArray
{

    MatrixStorage m_storage = MatrixStorage::FullMatrix;
    MatrixLayout m_layout = MatrixLayout::CStype;
    deg_t m_nrows = 0;
    deg_t m_ncols = 0;

public:
    ScalarMatrix();

    ScalarMatrix(const ScalarType* type, deg_t rows, deg_t cols,
                 MatrixStorage = MatrixStorage::FullMatrix,
                 MatrixLayout = MatrixLayout::CStype);

    ScalarMatrix(deg_t rows, deg_t cols, ScalarArray&& array,
                 MatrixStorage storage = MatrixStorage::FullMatrix,
                 MatrixLayout layout = MatrixLayout::CStype);

    RPY_NO_DISCARD constexpr deg_t nrows() const noexcept { return m_nrows; }

    RPY_NO_DISCARD constexpr deg_t ncols() const noexcept { return m_ncols; }

    RPY_NO_DISCARD constexpr MatrixStorage storage() const noexcept
    {
        return m_storage;
    }

    constexpr void storage(MatrixStorage new_storage)
    {
        // TODO: Check if this requires allocation or something
        m_storage = new_storage;
    }

    RPY_NO_DISCARD constexpr MatrixLayout layout() const noexcept
    {
        return m_layout;
    }

    constexpr void layout(MatrixLayout new_layout) noexcept
    {
        m_layout = new_layout;
    }

    RPY_NO_DISCARD
    ScalarMatrix row(deg_t i);
    RPY_NO_DISCARD
    ScalarMatrix row(deg_t i) const;
    RPY_NO_DISCARD
    ScalarMatrix col(deg_t i);
    RPY_NO_DISCARD
    ScalarMatrix col(deg_t i) const;

    RPY_NO_DISCARD
    ScalarPointer data() const;
    RPY_NO_DISCARD
    ScalarPointer data();

    RPY_NO_DISCARD
    ScalarMatrix to_full() const;
    RPY_NO_DISCARD
    ScalarMatrix to_full(MatrixLayout layout) const;
    void to_full(ScalarMatrix& into) const;

    RPY_SERIAL_SERIALIZE_FN();
};

RPY_SERIAL_SERIALIZE_FN_IMPL(ScalarMatrix)
{
    RPY_SERIAL_SERIALIZE_NVP("storage", m_storage);
    RPY_SERIAL_SERIALIZE_NVP("layout", m_layout);
    RPY_SERIAL_SERIALIZE_NVP("rows", m_nrows);
    RPY_SERIAL_SERIALIZE_NVP("cols", m_ncols);
    RPY_SERIAL_SERIALIZE_BASE(ScalarArray);
}

}// namespace scalars
}// namespace rpy

RPY_SERIAL_SPECIALIZE_TYPES(rpy::scalars::ScalarMatrix,
                            rpy::serial::specialization::member_serialize)

#endif// ROUGHPY_SCALARS_SCALAR_MATRIX_H_
