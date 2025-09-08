#include "cpu/dense_ft_fma.hpp"

#include "xla/ffi/api/ffi.h"
#include "nanobind/nanobind.h"

namespace {

namespace nb = nanobind;

// Validate function is an XLA FFI handler and return nanobind capsule
template <typename Fn>
nb::capsule encapsulate_handler(Fn* fn) {
    static_assert(
    std::is_invocable_r_v<XLA_FFI_Error*, Fn, XLA_FFI_CallFrame *>,
        "Encapsulated function must be an XLA FFI handler"
    );
    return nb::capsule(reinterpret_cast<void*>(fn));
}

} // namespace

NB_MODULE(_rpy_jax_internals, m) {
    // FIXME remove rms_norm
    m.def("rms_norm", []() { return encapsulate_handler(RmsNorm); });
    // registrations["cpu_dense_ft_fma"] = encapsulate_handler(cpu::dense_ft_fma);
}
