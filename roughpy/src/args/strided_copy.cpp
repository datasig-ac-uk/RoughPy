//
// Created by sam on 09/08/23.
//

#include "strided_copy.h"
#include <cstring>

void rpy::python::stride_copy(
        void* dst, const void* src, const py::ssize_t itemsize,
        const py::ssize_t ndim, const py::ssize_t* shape_in,
        const py::ssize_t* strides_in,
        const py::ssize_t* strides_out
) noexcept
{
    RPY_DBG_ASSERT(ndim == 1 || ndim == 2);

    auto* dptr = static_cast<char*>(dst);
    const auto* sptr = static_cast<const char*>(src);

    if (ndim == 1) {
        for (py::ssize_t i=0; i<shape_in[0]; ++i) {
            std::memcpy(dptr + i*strides_out[0],
                        sptr + i*strides_in[0],
                        itemsize);
        }
    } else {
        for (py::ssize_t i=0; i<shape_in[0]; ++i) {
            for (py::ssize_t j=0; j<shape_in[1]; ++j) {
                std::memcpy(
                        dptr + i*strides_in[0] + j*strides_in[1],
                        dptr + i*strides_out[0] + j*strides_out[1],
                        itemsize
                        );
            }
        }
    }

}
