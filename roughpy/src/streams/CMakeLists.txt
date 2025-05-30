set(SOURCES
    BaseStream.cpp
    BaseStream.h
    brownian_stream.cpp
    brownian_stream.h
    externally_sourced_stream.cpp
    externally_sourced_stream.h
    function_stream.cpp
    function_stream.h
    lie_increment_stream.h
    lie_increment_stream.cpp
    piecewise_abelian_stream.cpp
    piecewise_abelian_stream.h
    r_py_tick_construction_helper.cpp
    r_py_tick_construction_helper.h
    py_parametrization.cpp
    py_parametrization.h
    schema.cpp
    schema.h
    schema_finalization.cpp
    schema_finalization.h
    signature_arguments.cpp
    signature_arguments.h
    stream.cpp
    stream.h
    streams.cpp
    streams.h
    tensor_valued_stream.cpp
    tensor_valued_stream.h
    tick_stream.h
    tick_stream.cpp
)

target_sources(RoughPy_PyModule PRIVATE ${SOURCES})

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    target_sources(test_python_embed PRIVATE ${SOURCES})
endif ()
