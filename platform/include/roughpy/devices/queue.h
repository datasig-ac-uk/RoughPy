// Copyright (c) 2023 the RoughPy Developers. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROUGHPY_DEVICE_QUEUE_H_
#define ROUGHPY_DEVICE_QUEUE_H_

#include "core.h"
#include "device_object_base.h"

namespace rpy {
namespace devices {

/**
 * @class QueueInterface
 * @brief The QueueInterface class is an interface for a queue implementation.
 *
 * This class defines the public methods that must be implemented by any class
 * that wants to act as a queue.
 */
class ROUGHPY_DEVICES_EXPORT QueueInterface : public dtl::InterfaceBase
{
public:

    using object_t = Queue;

    virtual dimn_t size() const;
};

#ifdef RPY_PLATFORM_WINDOWS
#  ifdef RoughPy_Platform_EXPORTS
namespace dtl {
extern template class ObjectBase<QueueInterface, Queue>;
}
#  else
namespace dtl {
template class RPY_DLL_IMPORT ObjectBase<QueueInterface, Queue>;
}
#  endif
#else
namespace dtl {
extern template class ROUGHPY_DEVICES_EXPORT ObjectBase<QueueInterface, Queue>;
}
#endif

/**
 * @class Queue
 * @brief The Queue class represents a queue object.
 *
 * This class is derived from ObjectBase, which provides common functionalities
 * for device objects. The Queue class is used to store elements in a queue
 * data structure.
 */
class ROUGHPY_DEVICES_EXPORT Queue
    : public dtl::ObjectBase<QueueInterface, Queue>
{
    using base_t = dtl::ObjectBase<QueueInterface, Queue>;

public:
    using base_t::base_t;

    /**
     * @brief Returns the number of elements in the queue.
     *
     * This method returns the size of the queue by calling the size() method
     * of the implementation object, which must be available in order for
     * this method to work.
     *
     * @return The number of elements in the queue. If the implementation object
     *         is null, 0 is returned.
     */
    RPY_NO_DISCARD dimn_t size() const;

    /**
     * @brief Checks whether the queue is the default queue.
     *
     * This method checks whether the queue is the default queue. If the queue
     * is the default queue, it returns true; otherwise, it returns false.
     *
     * @return True if the queue is the default queue; otherwise, false.
     */
    RPY_NO_DISCARD bool is_default() const noexcept
    {
        return is_null();
    }
};



}// namespace device
}// namespace rpy

#endif// ROUGHPY_DEVICE_QUEUE_H_