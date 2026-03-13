#ifndef ROUGHPY_JAX_CPU_DENSE_ST_FMA_H
#define ROUGHPY_JAX_CPU_DENSE_ST_FMA_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cpu_dense_st_fma(XLA_FFI_CallFrame*);
XLA_FFI_Error* cpu_dense_st_mul(XLA_FFI_CallFrame*);


#ifdef __cplusplus
}
#endif
#endif  // ROUGHPY_JAX_CPU_DENSE_ST_FMA_H