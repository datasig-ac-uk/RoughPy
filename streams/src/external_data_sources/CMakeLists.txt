


target_sources(RoughPy_Streams PRIVATE
        csv_data_source.cpp
        csv_data_source.h
        sound_file_data_source.cpp
        sound_file_data_source.h
)

target_link_libraries(RoughPy_Streams PRIVATE SndFile::sndfile)
