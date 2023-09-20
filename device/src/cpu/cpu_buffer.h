//
// Created by sam on 20/09/23.
//

#ifndef ROUGHPY_CPU_BUFFER_H
#define ROUGHPY_CPU_BUFFER_H

#include <roughpy/device/core.h>
#include <roughpy/device/buffer.h>


namespace rpy {
namespace device {

class CPUBufferInterface : public BufferInterface
{

    struct Data {
        void * ptr;
        dimn_t n_bytes;
    };

public:

    static void* create_data(
            void* ptr, dimn_t bytes
            )
    {
        return new Data {ptr, bytes};
    }

    void* clone(void* content) const override;
    void clear(void* content) const override;
};


namespace cpu {

RPY_NO_DISCARD
const CPUBufferInterface* buffer_interface() noexcept;

}

}// namespace device
}// namespace rpy

#endif// ROUGHPY_CPU_BUFFER_H
