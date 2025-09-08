namespace {

namespace cpu = rpy::jax::cpu;

// Validate function is an XLA FFI handler and return nanobind capsule
template <typename Fn>
nb::capsule encapsulate_handler(Fn* fn) {
    static_assert(
    std::is_invocable_r_v<XLA_FFI_Error*, Fn, XLA_FFI_CallFrame *>,
        "Encapsulated function must be an XLA FFI handler"
    );
    return nb::capsule(reinterpret_cast<void*>(fn));
}

}

NB_MODULE(_roughpy_jax, m) {
    m.def("registrations", []() {
        nb::dict registrations;
        // registrations["cpu_dense_ft_fma"] = encapsulate_handler(cpu::dense_ft_fma);
        return registrations;
    });
}
