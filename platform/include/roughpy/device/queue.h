//
// Created by sammorley on 22/11/24.
//

#ifndef ROUGHPY_DEVICE_QUEUE_H
#define ROUGHPY_DEVICE_QUEUE_H

#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::device {


class ROUGHPY_PLATFORM_EXPORT Queue : public mem::RefCountedMiddle<>
{
public:

    Queue();

    virtual size_t size() const noexcept;

};

}


#endif //ROUGHPY_DEVICE_QUEUE_H
