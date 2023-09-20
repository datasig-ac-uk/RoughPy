//
// Created by sam on 20/09/23.
//

#include "cpu_buffer.h"
#include "cpu_device.h"


using namespace rpy;
using namespace rpy::device;
void* CPUBufferInterface::clone(void* content) const
{

}
void CPUBufferInterface::clear(void* content) const
{
    delete static_cast<Data*>(content);
}





const CPUBufferInterface* cpu::buffer_interface() noexcept
{
    static const CPUBufferInterface iface;
    return &iface;
}