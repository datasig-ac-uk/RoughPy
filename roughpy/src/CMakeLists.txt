target_sources(RoughPy_StaticLib PUBLIC
        roughpy_module.cpp
        roughpy_module.h
        roughpy_python.h
)

add_subdirectory(algebra)
add_subdirectory(args)
add_subdirectory(intervals)
add_subdirectory(scalars)
add_subdirectory(streams)

target_link_libraries(RoughPy_PyModule PRIVATE
        RoughPy_StaticLib
)
