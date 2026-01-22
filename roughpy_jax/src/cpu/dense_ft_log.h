#ifndef ROUGHPY_JAX_CPU_DENSE_FT_LOG_H
#define ROUGHPY_JAX_CPU_DENSE_FT_LOG_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cpu_dense_ft_log(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif  // ROUGHPY_JAX_CPU_DENSE_FT_LOG_H