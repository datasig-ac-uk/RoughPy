//
// Created by sam on 4/26/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
#define ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H

#include <roughpy/core/errors.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/types.h>

#include <roughpy/devices/type.h>

namespace rpy {
namespace devices {

namespace dtl {

template <typename T>
struct IDAndNameOfFType;

}

/**
 * @class FundamentalType
 * @brief A class that represents a fundamental data type.
 *
 * The FundamentalType class is a subclass of the Type class and represents a
 * fundamental data type. It provides methods to get the unique ID and name of
 * the type, as well as the type information and traits.
 */
template <typename T>
class RPY_LOCAL FundamentalType : public Type
{

public:
    FundamentalType(string_view id, string_view name)
        : Type(id, name, devices::type_info<T>(), devices::traits_of<T>())
    {}

    /**
     * @brief Returns a pointer to the static instance of FundamentalType<T>
     * with the specified type T.
     *
     * @tparam T The type for which the FundamentalType instance is returned.
     *
     * @return A pointer to the static instance of FundamentalType<T>.
     *
     * @note The returned pointer is valid throughout the program execution and
     * should not be deleted.
     * @note The returned pointer is guaranteed to be non-null.
     * @note The returned pointer may be used to access the ID and name of the
     * FundamentalType<T>.
     */
    RPY_NO_DISCARD static const FundamentalType* get() noexcept;
};

template <typename T>
const FundamentalType<T>* FundamentalType<T>::get() noexcept
{
    using IDName = dtl::IDAndNameOfFType<T>;
    static const FundamentalType type(IDName::id, IDName::name);
    return &type;
}
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_FINDAMENTAL_TYPE_H
