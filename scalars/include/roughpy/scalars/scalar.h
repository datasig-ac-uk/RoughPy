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

#ifndef ROUGHPY_SCALARS_SCALAR_H_
#define ROUGHPY_SCALARS_SCALAR_H_

#include "scalars_fwd.h"


#include "scalar_interface.h"
#include "scalar_pointer.h"
#include "scalar_type.h"

#include <roughpy/core/traits.h>
#include <roughpy/platform/serialization.h>
#include <roughpy/platform/archives.h>


namespace rpy {
namespace scalars {

class Scalar : private ScalarPointer
{

public:
    Scalar() = default;
    Scalar(const Scalar& other);
    Scalar(Scalar&& other) noexcept;
    explicit Scalar(const ScalarType* type);

    explicit Scalar(scalar_t arg);
    explicit Scalar(ScalarPointer ptr);
    explicit Scalar(ScalarInterface* interface_ptr);
    Scalar(ScalarPointer ptr, flags::PointerType ptype);
    Scalar(const ScalarType* type, scalar_t arg);

    Scalar(ScalarPointer other, uint32_t new_flags);

    template <typename I, typename J,
              typename = std::enable_if_t<std::is_integral<I>::value
                                          && std::is_integral<J>::value>>
    Scalar(const ScalarType* type, I numerator, J denominator)
    {
        if (type == nullptr) { type = get_type("rational"); }
        ScalarPointer::operator=(type->allocate(1));
        type->assign(static_cast<const ScalarPointer&>(*this),
                     static_cast<long long>(numerator),
                     static_cast<long long>(denominator));
    }

    template <typename ScalarArg>
    Scalar(const ScalarType* type, ScalarArg arg)
    {
        const auto* scalar_arg_type = ScalarType::of<ScalarArg>();
        if (scalar_arg_type != nullptr) {
            if (type == nullptr) { type = scalar_arg_type; }
            ScalarPointer::operator=(type->allocate(1));
            type->convert_copy(to_mut_pointer(),
                               {scalar_arg_type, std::addressof(arg)}, 1);
        } else {
            const auto& id = type_id_of<ScalarArg>();
            if (type == nullptr) { type = ScalarType::for_id(id); }
            ScalarPointer::operator=(type->allocate(1));
            type->convert_copy(to_mut_pointer(), {id, &arg}, 1);
        }
    }

    ~Scalar();

    Scalar& operator=(const Scalar& other);
    Scalar& operator=(Scalar&& other) noexcept;

    template <typename T,
              typename
              = std::enable_if_t<!std::is_same<Scalar, std::decay_t<T>>::value>>
    Scalar& operator=(T arg)
    {
        if (p_type == nullptr) {
            p_type = ScalarType::of<std::decay_t<T>>();
        } else {
            if (is_const()) {
                RPY_THROW(std::runtime_error,
                        "attempting to assign value to const scalar");
            }
        }

        if (p_data == nullptr) {
            m_flags |= flags::OwnedPointer;
            ScalarPointer::operator=(p_type->allocate(1));
        }

        if ((m_flags & interface_flag) != 0) {
            static_cast<ScalarInterface*>(const_cast<void*>(p_data))
                    ->assign(std::addressof(arg), type_id_of<T>());
        } else {
            const auto& type_id = type_id_of<T>();
            p_type->convert_copy(static_cast<ScalarPointer&>(*this),
                                 {type_id, std::addressof(arg)}, 1);

        }

        return *this;
    }

    using ScalarPointer::is_const;
    using ScalarPointer::type;
    RPY_NO_DISCARD
    bool is_value() const noexcept;
    RPY_NO_DISCARD
    bool is_zero() const noexcept;

    RPY_NO_DISCARD
    ScalarPointer to_pointer() const noexcept;
    RPY_NO_DISCARD
    ScalarPointer to_mut_pointer();

    void set_to_zero();

    RPY_NO_DISCARD
    scalar_t to_scalar_t() const;

    Scalar operator-() const;

    Scalar operator+(const Scalar& other) const;
    Scalar operator-(const Scalar& other) const;
    Scalar operator*(const Scalar& other) const;
    Scalar operator/(const Scalar& other) const;

    Scalar& operator+=(const Scalar& other);
    Scalar& operator-=(const Scalar& other);
    Scalar& operator*=(const Scalar& other);
    Scalar& operator/=(const Scalar& other);

    bool operator==(const Scalar& other) const noexcept;
    bool operator!=(const Scalar& other) const noexcept;

    // #ifndef RPY_DISABLE_SERIALIZATION
    // private:
    //     friend rpy::serialization_access;
    //
    //
    //     template <typename Ar>
    //     void save(Ar& ar, const unsigned int /*version*/) const {
    //         ar << get_type_id();
    //         if (is_interface()) {
    //             auto ptr = static_cast<const
    //             ScalarInterface*>(p_data)->to_pointer();
    //             // p_type cannot be null if the data is an interface
    //             ar << p_type->to_raw_bytes(ptr, 1);
    //         } else {
    //             ar << to_raw_bytes(1);
    //         }
    //     }
    //
    //     template <typename Ar>
    //     void load(Ar& ar, const unsigned int /*version*/) {
    //         std::string type_id;
    //         ar >> type_id;
    //         std::vector<byte> bytes;
    //         ar >> bytes;
    //         update_from_bytes(type_id, 1, bytes);
    //     }
    //
    // #endif

    RPY_SERIAL_SAVE_FN();
    RPY_SERIAL_LOAD_FN();
};

RPY_SERIAL_EXTERN_SAVE_CLS(Scalar)
RPY_SERIAL_EXTERN_LOAD_CLS(Scalar)

RPY_EXPORT
std::ostream& operator<<(std::ostream&, const Scalar& arg);




}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_H_
