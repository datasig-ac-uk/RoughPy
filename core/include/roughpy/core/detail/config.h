//
// Created by sam on 19/11/24.
//
#pragma once

#ifndef ROUGHPY_CORE_DETAIL_CONFIG_H
#  define ROUGHPY_CORE_DETAIL_CONFIG_H

// Detect operating system
#  if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
#    define RPY_PLATFORM_WINDOWS 1
#  elif defined(__unix__) || defined(__unix)
#    define RPY_PLATFORM_UNIX 1
#    if defined(__APPLE__) || defined(__MACH__)
#      define RPY_PLATFORM_MACOS 1
#    elif defined(__linux__)
#      define RPY_PLATFORM_LINUX 1
#    elif defined(__FreeBSD__)
#      define RPY_PLATFORM_FREEBSD 1
#    endif
#  elif defined(__APPLE__) || defined(__MACH__)
#    define RPY_PLATFORM_MACOS 1
#  else
#    define RPY_PLATFORM_UNKNOWN 0
#  endif

// Detect Architecture
#  if defined(_M_X64) || defined(__amd64__) || defined(__x86_64__)             \
          || defined(_M_AMD64)
#    define RPY_ARCH_X64
#  elif defined(_M_IX86) || defined(__i386__) || defined(_X86_)
#    define RPY_ARCH_X86
#  elif defined(__aarch64__) || defined(_M_ARM64)
#    define RPY_ARCH_ARM64
#  elif defined(__arm__) || defined(_M_ARM)
#    define RPY_ARCH_ARM
#  elif defined(__powerpc) || defined(_M_PPC) || defined(__ppc)                \
          || defined(_ARCH_PPC)
#    define RPY_ARCH_PPC
#  elif defined(__riscv)
#    define RPY_ARCH_RISCV
#  else
#    define RPY_ARCH_UNKNOWN
#  endif

// Detect Endianness
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define RPY_IS_LITTLE_ENDIAN 1
#elif defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define RPY_IS_BIG_ENDIAN 1
#elif RPY_PLATFORM_WINDOWS
// Abseil sets
#  define RPY_IS_LITTLE_ENDIAN 1
#else
#  error "Cannot detect the endianness of this platform"
#endif


// Detect Compiler and Version
#  if defined(_MSC_VER)
#    define RPY_COMPILER_VERSION(major, minor, patch) ((major)*10 + (minor))
#    define RPY_COMPILER_MSVC _MSC_VER
#  elif defined(__clang__)
#    define RPY_COMPILER_VERSION(major, minor, patch) \
       ((major)*10000 + (minor)*100 + (patch))
#    define RPY_COMPILER_CLANG \
      RPY_COMPILER_VERSION(__clang_major__, __clang_minor__, __clang_patchlevel__)
#  elif defined(__GNUC__) || defined(__GNUG__)
#    define RPY_COMPILER_VERSION(major, minor, patch) \
       ((major)*10000 + (minor)*100 + (patch))
#    define RPY_COMPILER_GCC                                                  \
        RPY_COMPILER_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#  elif defined(__INTEL_COMPILER)
#    define RPY_COMPILER_INTEL __INTEL_COMPILER
#  elif defined(__NVCC__)
#    define RPY_COMPILER_NVCC                                                  \
        (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100             \
         + __CUDACC_VER_PATCHLEVEL__)
#  else
#    define RPY_COMPILER_UNKNOWN 0
#  endif

// Detect the C++ standard
#  if defined(__cplusplus)
// At some point we might have non-c++ parts of the library.
#    if defined(_MSVC_LANG)
// MSVC does not correctly set __cplusplus, so we have to use _MSVC_LANG to get
// the version of the C++ standard supported by the compiler. (Abseil has a
// comment that even this is not always valid.)
#      define RPY_CPP_VERSION _MSVC_LANG
#    else
#      define RPY_CPP_VERSION __cplusplus
#    endif
#  endif

#  if RPY_CPP_VERSION >= 201103L
#    define RPY_CPP_11 1
#  endif
#  if RPY_CPP_VERSION >= 201403L
#    define RPY_CPP_14 1
#  endif
#  if RPY_CPP_VERSION >= 201703L
#    define RPY_CPP_17 1
#  endif
#  if RPY_CPP_VERSION >= 202002L
#    define RPY_CPP_20 1
#  endif
#  if RPY_CPP_VERSION >= 202303L
#    define RPY_CPP_23 1
#  endif

#endif// ROUGHPY_CORE_DETAIL_CONFIG_H
