// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
// Created by user on 09/05/23.
//

#ifndef ROUGHPY_PLATFORM_SERIALIZATION_H
#define ROUGHPY_PLATFORM_SERIALIZATION_H

#ifndef RPY_DISABLE_SERIALIZATION
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/strong_typedef.hpp>
#include <boost/serialization/assume_abstract.hpp>
#endif // RPY_DISABLE_SERIALIZATION

/*
 * For flexibility, and possibly later swapping out framework,
 * define redefine the relevant macros here as RPY_SERIAL_*.
 * pull in the access helper struct.
 */
#ifndef RPY_DISABLE_SERIALIZATION
#define RPY_STRONG_TYPEDEF(type, name) BOOST_STRONG_TYPEDEF(type, name)
#define RPY_SERIAL_SPLIT_MEMBER() BOOST_SERIALIZATION_SPLIT_MEMBER()
#define RPY_SERIAL_SPLIT_FREE(T) BOOST_SERIALIZATION_SPLIT_FREE(T)
#define RPY_SERIAL_ASSUME_ABSTRACT(T) BOOST_SERIALIZATION_ASSUME_ABSTRACT(T)
#else
#define RPY_STRONG_TYPEDEF(type, name) typedef type name
#define RPY_SERIAL_SPLIT_MEMBER()
#define RPY_SERIAL_SPLIT_FREE(T)
#define RPY_SERIAL_ASSUME_ABTRACT(T)
#endif



#ifndef RPY_DISABLE_SERIALIZATION
namespace rpy {

using serialization_access = boost::serialization::access;

namespace serial {

using boost::serialization::base_object;
using boost::serialization::make_array;


}
}

#endif // RPY_DISABLE_SERIALIZATION
#endif//ROUGHPY_PLATFORM_SERIALIZATION_H
