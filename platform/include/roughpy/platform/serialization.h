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
// Created by user on 09/05/23.
//

#ifndef ROUGHPY_PLATFORM_SERIALIZATION_H
#define ROUGHPY_PLATFORM_SERIALIZATION_H

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

class BinaryInputArchive;
class BinaryOutputArchive;
class JSONInputArchive;
class JSONOutputArchive;
class PortableBinaryInputArchive;
class PortableBinaryOutputArchive;
class XMLInputArchive;
class XMLOutputArchive;

}// namespace cereal

#include <roughpy/core/slice.h>

// #include "filesystem.h"

/*
 * For flexibility, and possibly later swapping out framework,
 * define redefine the relevant macros here as RPY_SERIAL_*.
 * pull in the access helper struct.
 */
#define RPY_SERIAL_ACCESS() friend class ::cereal::access
#define RPY_SERIAL_CLASS_VERSION(T, V) CEREAL_CLASS_VERSION(T, V)
#define RPY_SERIAL_SERIALIZE_VAL(V) archive(CEREAL_NVP(V))
#define RPY_SERIAL_SERIALIZE_SIZE(S) archive(::cereal::make_size_tag(S))
#define RPY_SERIAL_SERIALIZE_NVP(N, V) archive(::cereal::make_nvp(N, V))
#define RPY_SERIAL_SERIALIZE_BASE(B) archive(::cereal::base_class<B>(this))
#define RPY_SERIAL_SERIALIZE_BYTES(NAME, P, N)                                 \
    archive(::cereal::make_nvp(NAME, ::cereal::binary_data(P, N)))
#define RPY_SERIAL_SPECIALIZE_TYPES(T, M)                                      \
    CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(T, M)
#define RPY_SERIAL_SERIALIZE_BARE(V) archive(V)
#define RPY_SERIAL_REGISTER_CLASS(T) CEREAL_REGISTER_TYPE(T)
#define RPY_SERIAL_FORCE_DYNAMIC_INIT(LIB) CEREAL_FORCE_DYNAMIC_INIT(LIB)
#define RPY_SERIAL_DYNAMIC_INIT(LIB) CEREAL_REGISTER_DYNAMIC_INIT(LIB)
#define RPY_SERIAL_CLASS_RELATION(BASE, DERIVED)                               \
    CEREAL_REGISTER_POLYMORPHIC_RELATION(BASE, DERIVED)
#define RPY_SERIAL_REGISTER_TYPE_WITH_NAME(T, N)                               \
    CEREAL_REGISTER_TYPE_WITH_NAME(T, N)

#define RPY_SERIAL_LOAD_AND_CONSTRUCT(T)                                       \
    namespace cereal {                                                         \
    template <>                                                                \
    struct LoadAndConstruct<T> {                                               \
                                                                               \
        template <typename Archive>                                            \
        static void load_and_construct(                                        \
                Archive& archive,                                              \
                ::cereal::construct<T>& construct,                             \
                const std::uint32_t version                                    \
        );                                                                     \
    };                                                                         \
    }                                                                          \
                                                                               \
    template <typename Archive>                                                \
    void cereal::LoadAndConstruct<T>::load_and_construct(                      \
            Archive& archive,                                                  \
            ::cereal::construct<T>& construct,                                 \
            const std::uint32_t version                                        \
    )

#define RPY_SERIAL_SERIALIZE_FN()                                              \
    template <typename Archive>                                                \
    void serialize(Archive& archive, const std::uint32_t version)

#define RPY_SERIAL_LOAD_FN()                                                   \
    template <typename Archive>                                                \
    void load(Archive& archive, const std::uint32_t version)
#define RPY_SERIAL_SAVE_FN()                                                   \
    template <typename Archive>                                                \
    void save(Archive& archive, const std::uint32_t version) const

#define RPY_SERIAL_SERIALIZE_FN_IMPL(T)                                        \
    template <typename Archive>                                                \
    void T::serialize(                                                         \
            Archive& archive,                                                  \
            const std::uint32_t RPY_UNUSED_VAR version                         \
    )

#define RPY_SERIAL_LOAD_FN_IMPL(T)                                             \
    template <typename Archive>                                                \
    void T::load(Archive& archive, const std::uint32_t RPY_UNUSED_VAR version)

#define RPY_SERIAL_SAVE_FN_IMPL(T)                                             \
    template <typename Archive>                                                \
    void T::save(Archive& archive, const std::uint32_t RPY_UNUSED_VAR version) \
            const

#define RPY_SERIAL_SERIALIZE_FN_EXT(T)                                         \
    template <typename Archive>                                                \
    void serialize(                                                            \
            Archive& archive,                                                  \
            T& value,                                                          \
            const std::uint32_t RPY_UNUSED_VAR version                         \
    )

#define RPY_SERIAL_LOAD_FN_EXT(T)                                              \
    template <typename Archive>                                                \
    void load(                                                                 \
            Archive& archive,                                                  \
            T& value,                                                          \
            const std::uint32_t RPY_UNUSED_VAR version                         \
    )

#define RPY_SERIAL_SAVE_FN_EXT(T)                                              \
    template <typename Archive>                                                \
    void save(                                                                 \
            Archive& archive,                                                  \
            const T& value,                                                    \
            const std::uint32_t RPY_UNUSED_VAR version                         \
    )

#define RPY_SERIAL_EXT_LIB_SAVE_FN(T)                                          \
    namespace cereal {                                                         \
    template <typename Archive>                                                \
    void save(Archive& archive, const T& value);                               \
    }                                                                          \
    template <typename Archive>                                                \
    void cereal::save(Archive& archive, const T& value)

#define RPY_SERIAL_EXT_LIB_LOAD_FN(T)                                          \
    namespace cereal {                                                         \
    template <typename Archive>                                                \
    void load(Archive& archive, T& value);                                     \
    }                                                                          \
    template <typename Archive>                                                \
    void cereal::load(Archive& archive, T& value)

#define RPY_SERIAL_EXT_LIB_SERIALIZE_FN(T)                                     \
    namespace cereal {                                                         \
    template <typename Archive>                                                \
    void serialize(Archive& archive, T& value);                                \
    }                                                                          \
    template <typename Archive>                                                \
    void cereal::serialize(Archive& archive, T& value)

namespace rpy {
namespace serial {
using cereal::size_type;

}// namespace serial
}// namespace rpy

namespace rpy {

using serialization_access = cereal::access;

namespace serial {

using cereal::base_class;
using cereal::make_nvp;
using cereal::make_size_tag;

using cereal::specialization;

}// namespace serial

namespace archives {

using cereal::BinaryInputArchive;
using cereal::BinaryOutputArchive;
using cereal::JSONInputArchive;
using cereal::JSONOutputArchive;
using cereal::PortableBinaryInputArchive;
using cereal::PortableBinaryOutputArchive;
using cereal::XMLInputArchive;
using cereal::XMLOutputArchive;

}// namespace archives

#if defined(RPY_PLATFORM_WINDOWS)
#  define RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(CLASS, AR)                \
      extern template void CLASS::serialize<AR>(AR&, const std::uint32_t);

#  define RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(CLASS, AR)                     \
      extern template void CLASS::save<AR>(AR&, const std::uint32_t) const;

#  define RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(CLASS, AR)                     \
      extern template void CLASS::load<AR>(AR&, const std::uint32_t);

#  define RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(CLASS, AR)                  \
      template RPY_DLL_IMPORT void CLASS::serialize<AR>(                       \
              AR&,                                                             \
              const std::uint32_t                                              \
      );

#  define RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(CLASS, AR)                       \
      template RPY_DLL_IMPORT void CLASS::save<AR>(AR&, const std::uint32_t)   \
              const;

#  define RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(CLASS, AR)                       \
      template RPY_DLL_IMPORT void CLASS::load<AR>(AR&, const std::uint32_t);

#else

#  define RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(CLASS, AR)                \
      extern template void CLASS::serialize<AR>(AR&, const std::uint32_t);

#  define RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(CLASS, AR)                     \
      extern template void CLASS::save<AR>(AR&, const std::uint32_t) const;

#  define RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(CLASS, AR)                     \
      extern template void CLASS::load<AR>(AR&, const std::uint32_t);

#  define RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(CLASS, AR)                  \
      extern template void CLASS::serialize<AR>(AR&, const std::uint32_t);

#  define RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(CLASS, AR)                       \
      extern template void CLASS::save<AR>(AR&, const std::uint32_t) const;

#  define RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(CLASS, AR)                       \
      extern template void CLASS::load<AR>(AR&, const std::uint32_t);

#endif

#define RPY_SERIAL_EXTERN_SERIALIZE_CLS_BUILD(CLASS)                           \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::BinaryOutputArchive                               \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::JSONOutputArchive                                 \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryOutputArchive                       \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::XMLOutputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::BinaryInputArchive                                \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::JSONInputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryInputArchive                        \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_BUILD(                                \
            CLASS,                                                             \
            ::rpy::archives::XMLInputArchive                                   \
    )

#define RPY_SERIAL_EXTERN_SAVE_CLS_BUILD(CLASS)                                \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::BinaryOutputArchive                               \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::JSONOutputArchive                                 \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryOutputArchive                       \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::XMLOutputArchive                                  \
    )

#define RPY_SERIAL_EXTERN_LOAD_CLS_BUILD(CLASS)                                \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::BinaryInputArchive                                \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::JSONInputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryInputArchive                        \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_BUILD(                                     \
            CLASS,                                                             \
            ::rpy::archives::XMLInputArchive                                   \
    )

#define RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMP(CLASS)                             \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::BinaryOutputArchive                               \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::JSONOutputArchive                                 \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryOutputArchive                       \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::XMLOutputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::BinaryInputArchive                                \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::JSONInputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryInputArchive                        \
    )                                                                          \
    RPY_SERIAL_EXTERN_SERIALIZE_CLS_IMPL_IMP(                                  \
            CLASS,                                                             \
            ::rpy::archives::XMLInputArchive                                   \
    )

#define RPY_SERIAL_EXTERN_SAVE_CLS_IMP(CLASS)                                  \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::BinaryOutputArchive                               \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::JSONOutputArchive                                 \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryOutputArchive                       \
    )                                                                          \
    RPY_SERIAL_EXTERN_SAVE_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::XMLOutputArchive                                  \
    )

#define RPY_SERIAL_EXTERN_LOAD_CLS_IMP(CLASS)                                  \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::BinaryInputArchive                                \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::JSONInputArchive                                  \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(                                       \
            CLASS,                                                             \
            ::rpy::archives::PortableBinaryInputArchive                        \
    )                                                                          \
    RPY_SERIAL_EXTERN_LOAD_CLS_IMPL_IMP(CLASS, ::rpy::archives::XMLInputArchive)

using ByteSlice = Slice<byte>;

template <typename Archive, typename T>
enable_if_t<
        ::cereal::traits::
                is_output_serializable<::cereal::BinaryData<T>, Archive>::value
        && is_arithmetic_v<T>>
save(Archive& archive, const Slice<T>& data)
{
    archive(::cereal::binary_data(data.begin(), data.size()));
}

template <typename Archive, typename T>
enable_if_t<
        !::cereal::traits::
                is_output_serializable<::cereal::BinaryData<T>, Archive>::value
        || !is_arithmetic_v<T>>
save(Archive& archive, const Slice<T>& data)
{
    for (const auto& item : data) { archive(item); }
}

template <typename Archive, typename T>
enable_if_t<
        ::cereal::traits::
                is_input_serializable<::cereal::BinaryData<T>, Archive>::value
        && is_arithmetic_v<T>>
load(Archive& archive, Slice<T>& data)
{
    archive(::cereal::binary_data(data.begin(), data.size()));
}

template <typename Archive, typename T>
enable_if_t<
        !::cereal::traits::
                is_input_serializable<::cereal::BinaryData<T>, Archive>::value
        || !is_arithmetic_v<T>>
load(Archive& archive, Slice<T>& data)
{
    auto* ptr = data.begin();
    for (dimn_t i = 0; i < data.size(); ++i) { archive(ptr[i]); }
}

}// namespace rpy

#endif// ROUGHPY_PLATFORM_SERIALIZATION_H
