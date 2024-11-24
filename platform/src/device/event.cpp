//
// Created by sammorley on 22/11/24.
//


#include "roughpy/device/event.h"



using namespace rpy;
using namespace rpy::device;

Event::Event() : RefCountedMiddle()
{

}
void Event::wait()
{
    // do nothing
}
EventStatus Event::status() const noexcept
{
    return EventStatus::CompletedSuccessfully;
}
