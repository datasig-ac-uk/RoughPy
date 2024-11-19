//
// Created by sammorley on 19/11/24.
//

#ifndef ALLOC_H
#define ALLOC_H

#include "roughpy/core/macros.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::mem {
RPY_NO_DISCARD ROUGHPY_PLATFORM_EXPORT void* aligned_alloc(size_t alignment, size_t size);

ROUGHPY_PLATFORM_EXPORT void aligned_free(void* ptr);



}

#endif //ALLOC_H
