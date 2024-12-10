//
// Created by sammorley on 22/11/24.
//

#ifndef ROUGHPY_DEVICE_EVENT_H
#define ROUGHPY_DEVICE_EVENT_H


#include "roughpy/platform/reference_counting.h"
#include "roughpy/platform/roughpy_platform_export.h"

namespace rpy::device {


enum class EventStatus : uint8_t
{
    CompletedSuccessfully = 0,
    Pending = 1,
    Running = 2,
    Error = 3
};

class ROUGHPY_PLATFORM_EXPORT Event : public mem::RefCountedMiddle<>
{
public:

    Event();


    virtual void wait();
    virtual EventStatus status() const noexcept;
};


}

#endif //ROUGHPY_DEVICE_EVENT_H
