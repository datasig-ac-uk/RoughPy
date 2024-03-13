cmake_minimum_required(VERSION 3.21)


project(RoughPy_Algebra
        VERSION 0.0.1
        LANGUAGES CXX)

include(GNUInstallDirs)
include(GenerateExportHeader)




add_roughpy_component(Algebra
        SOURCES
        src/algebra.cpp
        src/basis.cpp
        src/basis_key.cpp
        src/context.cpp
        src/free_tensor.cpp
        src/hall_set_size.cpp
        src/hall_set_size.h
        src/key_range.cpp
        src/lie.cpp
        src/tensor_basis.cpp
        src/tensor_word.cpp
        src/tensor_word.h
        src/vector.cpp
        src/vector_iterator.cpp
        PUBLIC_HEADERS
        include/roughpy/algebra/algebra.h
        include/roughpy/algebra/algebra_fwd.h
        include/roughpy/algebra/basis.h
        include/roughpy/algebra/basis_key.h
        include/roughpy/algebra/context.h
        include/roughpy/algebra/context_fwd.h
        include/roughpy/algebra/linear_operator.h
        include/roughpy/algebra/vector.h
        include/roughpy/algebra/vector_iterator.h
        DEPENDENCIES PUBLIC
        Boost::boost
        NEEDS
        RoughPy::Core
        RoughPy::Platform
        RoughPy::Scalars
        )
