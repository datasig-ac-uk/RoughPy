//
// Created by sam on 3/30/24.
//

#ifndef FUNDAMENTAL_TYPE_H
#define FUNDAMENTAL_TYPE_H

#include "devices/core.h"
#include "devices/type.h"

namespace rpy {
namespace devices {

template <typename T>
class RPY_LOCAL FundamentalType : public Type
{

public:
    FundamentalType(string_view id, string_view name)
        : Type(id, name, devices::type_info<T>(), devices::traits_of<T>())
    {}
};

}// namespace devices
}// namespace rpy

#endif// FUNDAMENTAL_TYPE_H
