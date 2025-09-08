#include "cpu/dense_ft_fma.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cpu {

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T>& buffer)
{
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

ffi::Error RmsNormImpl(
    float eps,
    ffi::Buffer<ffi::F32> x,
    ffi::ResultBuffer<ffi::F32> y
) {
    auto [totalSize, lastDim] = GetDims(x);
    if (lastDim == 0) {
        return ffi::Error::InvalidArgument("RmsNorm input must be an array");
    }
    for (int64_t n = 0; n < totalSize; n += lastDim) {
        // FIXME replace with dense_ft_fma
        // ComputeRmsNorm(eps, lastDim, &(x.typed_data()[n]),
        // &(y->typed_data()[n]));
    }
    return ffi::Error::Success();
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm,
    rpy::jax::cpu::RmsNormImpl,
    xla::ffi::Ffi::Bind()
        .Attr<float>("eps")
        .Arg<ffi::Buffer<ffi::F32>>()  // x
        .Ret<ffi::Buffer<ffi::F32>>()  // y
);
