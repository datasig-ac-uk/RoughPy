


add_library(RoughPy_PyCore INTERFACE
        include/roughpy/pycore/compat.h
        include/roughpy/pycore/fnv1a_hash.h
        include/roughpy/pycore/object_handle.hpp
        include/roughpy/pycore/py_headers.h
)
add_library(RoughPy::PyCore ALIAS RoughPy_PyCore)

target_include_directories(RoughPy_PyCore INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(RoughPy_PyCore INTERFACE
        Python::Module
        Python::NumPy
        PythonCAPICompat
        RoughPy::Core
)


#target_sources(RoughPy_PyCore INTERFACE
#        include/roughpy/pycore/py_headers.h
#        include/roughpy/pycore/compat.h
#        include/roughpy/pycore/fnv1a_hash.h
#        include/roughpy/pycore/object_handle.hpp
#)

