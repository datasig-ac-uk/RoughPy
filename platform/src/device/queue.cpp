//
// Created by sammorley on 22/11/24.
//

#include "roughpy/device/queue.h"

using namespace rpy;
using namespace rpy::device;

Queue::Queue()
    : RefCountedMiddle()
{}
size_t Queue::size() const noexcept
{
    return 0;
}
