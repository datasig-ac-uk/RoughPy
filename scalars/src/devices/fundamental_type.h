//
// Created by sam on 3/30/24.
//

#ifndef FUNDAMENTAL_TYPE_H
#define FUNDAMENTAL_TYPE_H

#include "devices/core.h"
#include "devices/type.h"

namespace rpy {
namespace devices {

namespace dtl {

template <typename T>
struct IDAndNameOfFType;

}

template <typename T>
class RPY_LOCAL FundamentalType : public Type
{

public:
    FundamentalType(string_view id, string_view name)
        : Type(id, name, devices::type_info<T>(), devices::traits_of<T>())
    {}

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

#endif// FUNDAMENTAL_TYPE_H
