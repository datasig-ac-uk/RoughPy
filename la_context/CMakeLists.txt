cmake_minimum_required(VERSION 3.21)

project(RoughPy_LAContext
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)


if (NOT TARGET Libalgebra::Libalgebra)
    find_package(Libalgebra CONFIG REQUIRED)
endif ()


add_roughpy_lib(LAContext SHARED
        SOURCES
        src/la_context.cpp
        PUBLIC_HEADERS
        include/roughpy/la_context.h
        include/roughpy/la_context/free_tensor_info.h
        include/roughpy/la_context/lie_basis_info.h
        include/roughpy/la_context/lie_info.h
        include/roughpy/la_context/shuffle_tensor_info.h
        include/roughpy/la_context/tensor_basis_info.h
        include/roughpy/la_context/vector_iterator.h
        include/roughpy/la_context/vector_type_helper.h
        include/roughpy/la_context/vector_type_selector.h
        PUBLIC_DEPS
        RoughPy::Core
        RoughPy::Scalars
        RoughPy::Algebra
        Libalgebra::Libalgebra
        PVT_INCLUDE_DIR
        include/roughpy
        )


add_libalgebra_contexts(LAContext
        WIDTH 2 3 4 5 6 7 8
        CHECK_SIZE 2000000)
