set(SOURCES
    roughpy_module.cpp
    roughpy_module.h
    roughpy_python.h
)

target_sources(RoughPy_PyModule PRIVATE ${SOURCES})

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    target_sources(test_python_embed PRIVATE ${SOURCES})
endif ()


add_subdirectory(algebra)
add_subdirectory(args)
add_subdirectory(intervals)
add_subdirectory(scalars)
add_subdirectory(streams)