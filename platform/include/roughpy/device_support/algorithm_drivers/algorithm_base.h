//
// Created by sam on 5/17/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_ALGORITHM_BASE_H
#define ROUGHPY_DEVICE_SUPPORT_ALGORITHM_BASE_H

#include <roughpy/devices/type.h>

namespace rpy {
namespace devices {
namespace dtl {

template <typename S, typename T, bool>
struct AlgorithmFunctionBase {

    static void type_check(const Type* left, const Type* right)
    {
        RPY_CHECK(left->compare_with(right) == TypeComparison::AreSame);
    }
};

template <typename S, typename T>
struct AlgorithmFunctionBase<S, T, false> {
    static void type_check(const Type*, const Type*) {}
};

}// namespace dtl
}// namespace devices
}// namespace rpy

#endif// ROUGHPY_DEVICE_SUPPORT_ALGORITHM_BASE_H
