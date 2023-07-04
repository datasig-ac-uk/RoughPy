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
// Created by user on 28/02/23.
//

#include <roughpy/scalars/scalar_array.h>
#include <roughpy/scalars/scalar_type.h>

#include <roughpy/core/alloc.h>

using namespace rpy;
using namespace rpy::scalars;

rpy::scalars::ScalarArray::ScalarArray(
        rpy::scalars::ScalarArray&& other) noexcept
    : ScalarPointer(static_cast<ScalarPointer&&>(other)), m_size(other.m_size)
{
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
rpy::scalars::ScalarArray&
rpy::scalars::ScalarArray::operator=(rpy::scalars::ScalarArray&& other) noexcept
{
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
rpy::scalars::ScalarArray rpy::scalars::ScalarArray::borrow() const noexcept
{
    auto flag = m_flags & ~owning_flag;
    flag |= flags::IsConst;
    return {{p_type, p_data, flag}, m_size};
}
rpy::scalars::ScalarArray rpy::scalars::ScalarArray::borrow_mut() noexcept
{
    auto flag = m_flags & ~owning_flag;
    return {{p_type, p_data, flag}, m_size};
}

//
// BOOST_CLASS_EXPORT_IMPLEMENT(rpy::scalars::ScalarArray)
//
//
//
// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>
//
// template void ScalarArray::serialize(boost::archive::text_oarchive& ar, const
// unsigned int version); template void
// ScalarArray::serialize(boost::archive::text_iarchive& ar, const unsigned int
// version);

#define RPY_SERIAL_IMPL_CLASSNAME rpy::scalars::ScalarArray
#define RPY_SERIAL_DO_SPLIT

#include <roughpy/platform/serialization_instantiations.inl>
