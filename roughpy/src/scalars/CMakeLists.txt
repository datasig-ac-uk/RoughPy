set(SOURCES
    pytype_conversion.cpp
    pytype_conversion.h
    r_py_polynomial.cpp
    r_py_polynomial.h
    scalar_type.h
    scalar_type.cpp
    scalars.cpp
    scalars.h
)

target_sources(RoughPy_PyModule PRIVATE ${SOURCES})

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    target_sources(test_python_embed PRIVATE ${SOURCES})
endif ()
