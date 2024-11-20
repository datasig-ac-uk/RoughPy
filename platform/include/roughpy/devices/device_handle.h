//
// Created by sam on 20/11/24.
//

#ifndef ROUGHPY_PLATFORM_DEVICE_HANDLE_H
#define ROUGHPY_PLATFORM_DEVICE_HANDLE_H

#include <atomic>

#include "roughpy/core/check.h"
#include "roughpy/core/debug_assertion.h"
#include "roughpy/core/macros.h"
#include "roughpy/core/smart_ptr.h"
#include "roughpy/core/traits.h"
#include "roughpy/core/types.h"

#include "roughpy/platform/alloc.h"
#include "roughpy/platform/roughpy_platform_export.h"


namespace rpy {

namespace generics {

class Type;

}



namespace devices {


class Memory;

class ROUGHPY_PLATFORM_EXPORT DeviceHandle : public mem::SmallObjectBase
{
    mutable std::atomic<intptr_t> m_ref_count;


public:

    virtual ~DeviceHandle();

protected:

    DeviceHandle() noexcept;

    virtual void inc_ref() const noexcept;

    RPY_NO_DISCARD
    virtual bool dec_ref() const noexcept;

public:

    RPY_NO_DISCARD
    virtual intptr_t ref_count() const noexcept;

    friend void intrusive_ptr_inc_ref(const DeviceHandle* ptr) noexcept
    {
        ptr->inc_ref();
    }
    friend void intrusive_ptr_release(const DeviceHandle* ptr) noexcept
    {
        if (ptr->dec_ref()) {
            delete ptr;
        }
    }


    RPY_NO_DISCARD virtual Rc<generics::Memory>
    allocate_memory(const generics::Type& type, size_t size) const = 0;





    static Rc<const DeviceHandle> host() noexcept;

};




}
}

#endif //ROUGHPY_PLATFORM_DEVICE_HANDLE_H
