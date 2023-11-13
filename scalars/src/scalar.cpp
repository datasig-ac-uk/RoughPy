

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

#include <roughpy/scalars/scalar.h>
#include <roughpy/scalars/scalar_interface.h>
#include <roughpy/scalars/scalar_type.h>
#include <roughpy/scalars/traits.h>

#include "scalar/arithmetic.h"
#include "scalar/casts.h"
#include "scalar/comparison.h"
#include <roughpy/device/types.h>

#include <stdexcept>

using namespace rpy;
using namespace rpy::scalars;

void Scalar::allocate_data() {
    RPY_DBG_ASSERT(p_type_and_content_type.is_pointer());
    opaque_pointer = p_type_and_content_type->allocate_single();
}

Scalar::Scalar(Scalar::type_pointer type) : integer_for_convenience(0)
{
    if (!type.is_null()) {
        dtl::ScalarContentType mode;
        if (type.is_pointer()) {
            mode = dtl::content_type_of(type);
            p_type_and_content_type = type_pointer(type.get_pointer(), mode);
        } else {
            auto info = type.get_type_info();
            mode = dtl::content_type_of(info);
            if (mode == dtl::ScalarContentType::OwnedPointer) {
                p_type_and_content_type
                        = type_pointer(*scalar_type_of(info), mode);
            } else {
                p_type_and_content_type = type_pointer(info, mode);
            }
        }

        if (mode == dtl::ScalarContentType::OwnedPointer) {
            opaque_pointer = type->allocate_single();
        }
    }
}

Scalar::Scalar(const ScalarType* type)
    : p_type_and_content_type(type, dtl::content_type_of(type->type_info())),
      integer_for_convenience(0)
{
    if (p_type_and_content_type.get_enumeration()
        == dtl::ScalarContentType::OwnedPointer) {
        opaque_pointer = type->allocate_single();
    }
}
Scalar::Scalar(devices::TypeInfo info)
    : p_type_and_content_type(info, dtl::content_type_of(info)),
      integer_for_convenience(0)
{
    auto mode = p_type_and_content_type.get_enumeration();
    if (mode == dtl::ScalarContentType::OwnedPointer) {
        p_type_and_content_type = type_pointer(*scalar_type_of(info), mode);
        opaque_pointer = p_type_and_content_type->allocate_single();
    }
}
Scalar::Scalar() : p_type_and_content_type(), integer_for_convenience(0) {}

Scalar::Scalar(const ScalarType* type, void* ptr)
    : p_type_and_content_type(type, dtl::ScalarContentType::OpaquePointer),
      opaque_pointer(ptr)
{}
Scalar::Scalar(const ScalarType* type, const void* ptr)
    : p_type_and_content_type(type, dtl::ScalarContentType::ConstOpaquePointer),
      opaque_pointer(const_cast<void*>(ptr))
{}

Scalar::Scalar(devices::TypeInfo info, void* ptr)
    : p_type_and_content_type(info, dtl::ScalarContentType::OpaquePointer),
      opaque_pointer(ptr)
{}
Scalar::Scalar(devices::TypeInfo info, const void* ptr)
    : p_type_and_content_type(info, dtl::ScalarContentType::ConstOpaquePointer),
      opaque_pointer(const_cast<void*>(ptr))
{}
Scalar::Scalar(const ScalarType* type, int64_t num, int64_t denom)
{
    auto info = type->type_info();
    if (traits::is_arithmetic(info) && info.bytes <= sizeof(void*)) {
        p_type_and_content_type = type_pointer(type, dtl::ScalarContentType::TrivialBytes);
        dtl::scalar_assign_rational(trivial_bytes, info, num, denom);
    } else {
        p_type_and_content_type = type_pointer(type, dtl::ScalarContentType::OwnedPointer);
        opaque_pointer = type->allocate_single();
        dtl::scalar_assign_rational(opaque_pointer, info, num, denom);
    }
}

Scalar::Scalar(const Scalar& other) {}
Scalar::Scalar(Scalar&& other) noexcept {}

Scalar::~Scalar()
{
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::OwnedPointer:
            RPY_CHECK(p_type_and_content_type.is_pointer());
            p_type_and_content_type->free_single(opaque_pointer);
            break;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: interface.~unique_ptr();
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::ConstOpaquePointer: break;
    }
};

Scalar& Scalar::operator=(const Scalar& other)
{
    if (&other != this) {
        if (this->fast_is_zero()) {
            p_type_and_content_type = other.p_type_and_content_type;

            switch (p_type_and_content_type.get_enumeration()) {
                case dtl::ScalarContentType::TrivialBytes:
                case dtl::ScalarContentType::ConstTrivialBytes:
                    std::memcpy(
                            trivial_bytes,
                            other.trivial_bytes,
                            sizeof(interface_pointer_t)
                    );
                    break;
                case dtl::ScalarContentType::OpaquePointer:
                case dtl::ScalarContentType::ConstOpaquePointer:
                    opaque_pointer = other.opaque_pointer;
                    break;
                case dtl::ScalarContentType::OwnedPointer:
                    // TODO: implement copy for owned pointers
                    break;
                case dtl::ScalarContentType::Interface:
                case dtl::ScalarContentType::OwnedInterface:
                    // Copying of interface pointers is disallowed.
                    RPY_THROW(
                            std::runtime_error,
                            "copying of interface pointers "
                            "is not allowed"
                    );
            }
        } else {
            dtl::scalar_convert_copy(
                    mut_pointer(),
                    p_type_and_content_type.get_type_info(),
                    other
            );
        }
    }
    return *this;
}

Scalar& Scalar::operator=(Scalar&& other) noexcept
{
    if (&other != this) {
        this->~Scalar();
        p_type_and_content_type = other.p_type_and_content_type;
        switch (p_type_and_content_type.get_enumeration()) {
            case dtl::ScalarContentType::TrivialBytes:
            case dtl::ScalarContentType::ConstTrivialBytes:
                std::memcpy(
                        trivial_bytes,
                        other.trivial_bytes,
                        sizeof(interface_pointer_t)
                );
                break;
            case dtl::ScalarContentType::OpaquePointer:
            case dtl::ScalarContentType::ConstOpaquePointer:
            case dtl::ScalarContentType::OwnedPointer:
                opaque_pointer = other.opaque_pointer;
                other.opaque_pointer = nullptr;
                break;
            case dtl::ScalarContentType::Interface:
            case dtl::ScalarContentType::OwnedInterface:
                interface = std::move(other.interface);
                other.interface = nullptr;
                break;
        }
    }
    return *this;
}


#define DO_OP_FOR_ALL_TYPES(INFO) \
    switch(INFO.code) {\
        casse devices::TypeCode::Int: \
            switch (INFO.bytes) { \
                case \





bool Scalar::is_zero() const noexcept
{
    if (fast_is_zero()) { return true; }

    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
            return bit_cast<uintptr_t>(trivial_bytes) == 0;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer:
            return is_pointer_zero(opaque_pointer, p_type_and_content_type);
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            return is_pointer_zero(
                    interface->pointer(),
                    p_type_and_content_type
            );
    }

    RPY_UNREACHABLE_RETURN(false);
}

bool Scalar::is_reference() const noexcept
{
    if (fast_is_zero()) { return false; }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::OwnedPointer: return false;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: return true;
    }
    RPY_UNREACHABLE_RETURN(false);
}

bool Scalar::is_const() const noexcept
{
    if (fast_is_zero()) { return true; }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes: return false;
        case dtl::ScalarContentType::ConstTrivialBytes: return true;
        case dtl::ScalarContentType::OpaquePointer: return false;
        case dtl::ScalarContentType::ConstOpaquePointer: return true;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface: return false;
        case dtl::ScalarContentType::OwnedPointer: return false;
    }
    RPY_UNREACHABLE_RETURN(false);
}

const void* Scalar::pointer() const noexcept
{
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes: return &trivial_bytes;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer: return opaque_pointer;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            return interface->pointer();
    }
    RPY_UNREACHABLE_RETURN(nullptr);
}
void* Scalar::mut_pointer()
{
    if (fast_is_zero()) {
        RPY_THROW(
                std::runtime_error,
                "cannot get mutable pointer to constant"
                " value zero"
        );
    }
    switch (p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes: return &trivial_bytes;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::OwnedPointer: return opaque_pointer;
        case dtl::ScalarContentType::ConstTrivialBytes:
        case dtl::ScalarContentType::ConstOpaquePointer:
            RPY_THROW(
                    std::runtime_error,
                    "cannot get mutable pointer to constant value"
            );
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            RPY_THROW(
                    std::runtime_error,
                    "cannot get mutable pointer to special scalar references"
            );
    }

    RPY_UNREACHABLE_RETURN(nullptr);
}
optional<const ScalarType*> Scalar::type() const noexcept
{
    if (p_type_and_content_type.is_pointer()) {
        return p_type_and_content_type.get_pointer();
    }

    auto info = p_type_and_content_type.get_type_info();
    return  scalar_type_of(info);
}
devices::TypeInfo Scalar::type_info() const noexcept
{
    if (p_type_and_content_type.is_pointer()) {
        return p_type_and_content_type->type_info();
    }
    return p_type_and_content_type.get_type_info();
}

static void print_trivial_bytes(
        std::ostream& os,
        const byte* bytes,
        PackedScalarTypePointer<scalars::dtl::ScalarContentType> p_type
)
{}

static void print_opaque_pointer(
        std::ostream& os,
        const void* opaque,
        PackedScalarTypePointer<scalars::dtl::ScalarContentType> p_type
)
{}

std::ostream& rpy::scalars::operator<<(std::ostream& os, const Scalar& value)
{
    if (value.fast_is_zero()) {
        os << 0;
        return os;
    }

    switch (value.p_type_and_content_type.get_enumeration()) {
        case dtl::ScalarContentType::TrivialBytes:
        case dtl::ScalarContentType::ConstTrivialBytes:
            print_trivial_bytes(
                    os,
                    value.trivial_bytes,
                    value.p_type_and_content_type
            );
            break;
        case dtl::ScalarContentType::OpaquePointer:
        case dtl::ScalarContentType::ConstOpaquePointer:
        case dtl::ScalarContentType::OwnedPointer:
            print_opaque_pointer(
                    os,
                    value.opaque_pointer,
                    value.p_type_and_content_type
            );
            break;
        case dtl::ScalarContentType::Interface:
        case dtl::ScalarContentType::OwnedInterface:
            value.interface->print(os);
            break;
    }

    return os;
}
