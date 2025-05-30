set(SOURCES
    date_time_interval.cpp
    date_time_interval.h
    dyadic.h
    dyadic.cpp
    dyadic_interval.cpp
    dyadic_interval.h
    interval.h
    interval.cpp
    intervals.cpp
    intervals.h
    partition.cpp
    partition.h
    real_interval.cpp
    real_interval.h
    segmentation.cpp
    segmentation.h
)

target_sources(RoughPy_PyModule PRIVATE ${SOURCES})

if (ROUGHPY_BUILD_TEST_PYTHON_EMBED)
    target_sources(test_python_embed PRIVATE ${SOURCES})
endif ()
