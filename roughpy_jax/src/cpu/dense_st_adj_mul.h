#ifndef ROUGHPY_JAX_SRC_CPU_DENSE_ST_ADJ_MUL_HPP
#define ROUGHPY_JAX_SRC_CPU_DENSE_ST_ADJ_MUL_HPP

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif


XLA_FFI_Error* cpu_dense_st_adj_mul(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif


#endif// ROUGHPY_JAX_SRC_CPU_DENSE_ST_ADJ_MUL_HPP
