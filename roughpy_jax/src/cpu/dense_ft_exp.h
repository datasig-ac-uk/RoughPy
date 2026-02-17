#ifndef ROUGHPY_JAX_CPU_DENSE_FT_EXP_H
#define ROUGHPY_JAX_CPU_DENSE_FT_EXP_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cpu_dense_ft_exp(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif  // ROUGHPY_JAX_CPU_DENSE_FT_EXP_H