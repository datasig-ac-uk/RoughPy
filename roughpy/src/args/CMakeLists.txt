

target_sources(RoughPy_PyModule PRIVATE
        buffer_info.cpp
        buffer_info.h
        check_for_excess_args.cpp
        convert_timestamp.cpp
        convert_timestamp.h
        dlpack_helpers.cpp
        dlpack_helpers.h
        kwargs_to_path_metadata.h
        kwargs_to_path_metadata.cpp
        kwargs_to_vector_construction.cpp
        kwargs_to_vector_construction.h
        numpy.cpp
        numpy.h
        parse_algebra_configuration.cpp
        parse_algebra_configuration.h
        parse_data_argument.cpp
        parse_data_argument.h
        parse_schema.cpp
        parse_schema.h
        strided_copy.cpp
        strided_copy.h
)