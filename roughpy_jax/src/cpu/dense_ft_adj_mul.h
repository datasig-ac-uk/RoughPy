#ifndef ROUGHPY_JAX_SRC_CPU_DENSE_FT_ADJ_MUL_HPP
#define ROUGHPY_JAX_SRC_CPU_DENSE_FT_ADJ_MUL_HPP


#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif


XLA_FFI_Error* cpu_dense_ft_adj_lmul(XLA_FFI_CallFrame*);
XLA_FFI_Error* cpu_dense_ft_adj_rmul(XLA_FFI_CallFrame*);



#ifdef  __cplusplus
}
#endif


#endif// ROUGHPY_JAX_SRC_CPU_DENSE_FT_ADJ_MUL_HPP
