//
// Created by sam on 23/10/23.
//

#ifndef ROUGHPY_CUDA_BUFFER_H
#define ROUGHPY_CUDA_BUFFER_H

#include <roughpy/device/buffer.h>
#include "cuda_decls.h"

namespace rpy {
namespace devices {

class CUDABuffer : public BufferInterface
{
    void* p_data;
    CUDADevice m_device;


public:
    virtual BufferMode mode() const;
    virtual dimn_t size() const;
    virtual void* ptr();
    virtual ~CUDABuffer();
    virtual DeviceType type() const noexcept;
    virtual dimn_t ref_count() const noexcept;
    virtual std::unique_ptr<InterfaceBase> clone() const;
    virtual Device device() const noexcept;
};

}// namespace devices
}// namespace rpy

#endif// ROUGHPY_CUDA_BUFFER_H
