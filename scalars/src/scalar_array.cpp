// Copyright (c) 2023 RoughPy Developers. All rights reserved.
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
// Created by user on 28/02/23.
//

#include "scalar_array.h"

#include "scalar_type.h"

rpy::scalars::ScalarArray::ScalarArray(const rpy::scalars::ScalarType *type, rpy::dimn_t size)
    : ScalarPointer(type), m_size(0)
{
    if (p_type != nullptr && size > 0) {
        ScalarPointer::operator=(p_type->allocate(size));
        m_size = size;
        m_flags |= flags::OwnedPointer;
    }
}

rpy::scalars::ScalarArray::~ScalarArray() {
    if (p_type != nullptr && p_data != nullptr && is_owning()) {
        p_type->free(static_cast<ScalarPointer&>(*this), m_size);
    }
}

rpy::scalars::ScalarArray::ScalarArray(rpy::scalars::ScalarArray &&other) noexcept
    : ScalarPointer(static_cast<ScalarPointer&&>(other)), m_size(other.m_size) {
    /*
     * It doesn't really matter for this class, but various
     * derived classes will need to make sure that ownership
     * is transferred on move. We reset the p_data and size
     * to null values so that, in derived classes, the
     * destructor does not free the data while it is still
     * in use.
     */
    other.p_data = nullptr;
    other.m_size = 0;
}
rpy::scalars::ScalarArray &rpy::scalars::ScalarArray::operator=(rpy::scalars::ScalarArray &&other) noexcept {
    if (std::addressof(other) != this) {
        this->~ScalarArray();

        p_type = other.p_type;
        p_data = other.p_data;
        m_size = other.m_size;

        /*
         * It doesn't really matter for this class, but various
         * derived classes will need to make sure that ownership
         * is transferred on move. We reset the p_data and size
         * to null values so that, in derived classes, the
         * destructor does not free the data while it is still
         * in use.
         */
        other.p_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}
