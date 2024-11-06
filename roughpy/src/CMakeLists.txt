


target_sources(RoughPy_PyModule PRIVATE
        roughpy_module.cpp
        roughpy_module.h
        roughpy_python.h
)

add_subdirectory(algebra)
add_subdirectory(args)
add_subdirectory(intervals)
add_subdirectory(scalars)
add_subdirectory(streams)