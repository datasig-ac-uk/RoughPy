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
class FundamentalType : public Type {

public:

    FundamentalType(string id, string name)
        : Type(std::move(id), std::move(name), type_info<T>(), traits_of<T>())
    {}

};

}}

#endif //FUNDAMENTAL_TYPE_H
