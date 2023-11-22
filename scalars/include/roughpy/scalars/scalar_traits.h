#ifndef ROUGHPY_SCALARS_SCALAR_TRAITS_H_
#define ROUGHPY_SCALARS_SCALAR_TRAITS_H_

#include "scalar_interface.h"
#include "scalars_fwd.h"
#include "scalar.h"
#include "scalar_types.h"

namespace rpy {
namespace scalars {

template <typename T>
class scalar_type_trait
{
public:
    using value_type = T;
    using rational_type = T;
    using reference = T&;
    using const_reference = const T&;

    RPY_NO_DISCARD static optional<const ScalarType*> get_type() noexcept
    {
        return scalar_type_of<T>();
    }

    RPY_NO_DISCARD
    static Scalar make(value_type&& arg) {
        auto type = get_type();
        if (!type) {
            RPY_THROW(std::runtime_error, "not a valid scalar");
        }
        return Scalar(*type, std::move(arg));
    }
};


template <typename T>
class scalar_type_trait<T&> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T&;
    using const_reference = const T&;

    RPY_NO_DISCARD static optional<const ScalarType*> get_type() noexcept
    {
        return scalar_type_of<T>();
    }

    RPY_NO_DISCARD
    static Scalar make(reference arg) {
        auto type = get_type();
        if (!type) {
            RPY_THROW(std::runtime_error, "not a valid scalar");
        }
        return Scalar(*type, &arg);
    }

};

template <typename T>
class scalar_type_trait<const T&> {
public:
    using value_type = T;
    using rational_type = T;
    using reference = T&;
    using const_reference = const T&;

    RPY_NO_DISCARD static optional<const ScalarType*> get_type() noexcept
    {
        return scalar_type_of<T>();
    }

    RPY_NO_DISCARD
    static Scalar make(const_reference arg) {
        auto type = get_type();
        if (!type) {
            RPY_THROW(std::runtime_error, "not a valid scalar");
        }
        return Scalar(*type, &arg);
    }

};

}// namespace scalars
}// namespace rpy

#endif// ROUGHPY_SCALARS_SCALAR_TRAITS_H_
