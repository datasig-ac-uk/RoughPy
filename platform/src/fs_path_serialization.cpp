//
// Created by sam on 27/09/23.
//


#include <roughpy/platform/filesystem.h>
#include <roughpy/platform/serialization.h>

#define RPY_SERIAL_EXTERNAL rpy::fs
#define RPY_SERIAL_IMPL_CLASSNAME path
#define RPY_SERIAL_DO_SPLIT
#include <roughpy/platform/serialization_instantiations.inl>
