

set(INCLUDE_ROOT "${RoughPy_Platform_SOURCE_DIR}/include/roughpy/")

target_sources(RoughPy_Platform PRIVATE
        ${INCLUDE_ROOT}/platform/memory.h
        aligned_alloc.cpp
        small_object_base.cpp
        AlignedMemory.cpp
        AlignedMemory.h
        small_object_memory.cpp
        small_object_alloc.cpp
        small_object_memory.h
)



