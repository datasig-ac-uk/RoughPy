

add_library(RoughPy_OCL_Kernels STATIC)


target_sources(RoughPy_OCL_Kernels PRIVATE
        kernel_types.h
        masked_unary.c
        masked_binary.c
        free_tensor_multiply.c
)
