// Copyright (c) 2023 Datasig Developers. All rights reserved.
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
#ifndef RPY_DISABLE_SERIALIZATION
#  ifndef RPY_SERIAL_IMPL_CLASSNAME
#    error "class name not set"
#  endif

#  ifdef RPY_SERIAL_EXTERNAL
#    define ADD_ARCHIVE(ARCHIVE)                                               \
        namespace RPY_SERIAL_EXTERNAL {                                        \
        template void serialize(ARCHIVE&, RPY_SERIAL_IMPL_CLASSNAME&,          \
                                const std::uint32_t);                          \
        }
#    define ADD_LOAD(ARCHIVE)                                                  \
        namespace RPY_SERIAL_EXTERNAL {                                        \
        template void load(ARCHIVE&, RPY_SERIAL_IMPL_CLASSNAME&,               \
                           const std::uint32_t);                               \
        }
#    define ADD_SAVE(ARCHIVE)                                                  \
        namespace RPY_SERIAL_EXTERNAL {                                        \
        template void save(ARCHIVE&, const RPY_SERIAL_IMPL_CLASSNAME&,         \
                           const std::uint32_t);                               \
        }
#  else
#    define ADD_ARCHIVE(ARCHIVE)                                               \
        template void RPY_SERIAL_IMPL_CLASSNAME::serialize(                    \
                ARCHIVE& ar, const std::uint32_t version)
#    define ADD_LOAD(ARCHIVE)                                                  \
        template void RPY_SERIAL_IMPL_CLASSNAME::load(                         \
                ARCHIVE& ar, const std::uint32_t version)
#    define ADD_SAVE(ARCHIVE)                                                  \
        template void RPY_SERIAL_IMPL_CLASSNAME::save(                         \
                ARCHIVE& ar, const std::uint32_t version) const
#  endif

#  if defined(RPY_SERIAL_DO_SPLIT)
ADD_SAVE(::cereal::BinaryOutputArchive);
ADD_LOAD(::cereal::BinaryInputArchive);
ADD_SAVE(::cereal::PortableBinaryOutputArchive);
ADD_LOAD(::cereal::PortableBinaryInputArchive);
ADD_SAVE(::cereal::JSONOutputArchive);
ADD_LOAD(::cereal::JSONInputArchive);
#  else
ADD_ARCHIVE(::cereal::BinaryOutputArchive);
ADD_ARCHIVE(::cereal::BinaryInputArchive);
ADD_ARCHIVE(::cereal::PortableBinaryOutputArchive);
ADD_ARCHIVE(::cereal::PortableBinaryInputArchive);
ADD_ARCHIVE(::cereal::JSONOutputArchive);
ADD_ARCHIVE(::cereal::JSONInputArchive);
#  endif

#  undef ADD_ARCHIVE
#  undef ADD_LOAD
#  undef ADD_SAVE
#  undef RPY_SERIAL_DO_SPLIT
#  undef RPY_SERIAL_IMPL_CLASSNAME
#  undef RPY_SERIAL_EXTERNAL
#endif
