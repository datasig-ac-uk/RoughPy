//
// Created by sam on 6/12/24.
//

#ifndef ROUGHPY_DEVICE_SUPPORT_MACROS_H
#define ROUGHPY_DEVICE_SUPPORT_MACROS_H

#include <roughpy/core/macros.h>

#ifndef RPY_HOST
#  define RPY_HOST
#endif

#ifndef RPY_DEVICE
#  define RPY_DEVICE
#endif

#ifndef RPY_HOST_DEVICE
#  define RPY_HOST_DEVICE RPY_HOST RPY_DEVICE
#endif

#ifndef RPY_KERNEL
#  define RPY_KERNEL
#endif

#endif// ROUGHPY_DEVICE_SUPPORT_MACROS_H
