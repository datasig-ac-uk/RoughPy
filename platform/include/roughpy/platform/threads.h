//
// Created by sam on 28/07/23.
//

#ifndef ROUGHPY_PLATFORM_THREADS_H
#define ROUGHPY_PLATFORM_THREADS_H

#include <roughpy/core/types.h>
#include <roughpy/core/macros.h>
#include <roughpy/core/alloc.h>


#if defined(_OPENMP)
#define RPY_THREADING_OPENMP 1
#endif


namespace rpy { namespace platform {

enum struct ThreadBackend {
    Disabled=0,
    OpenMP=1
};



struct ThreadState {
    /// The threading library responsible for managing multi-threaded
    /// computation.
    ThreadBackend backend;

    /// The maximum number of threads that can be spawned by a single operation.
    int max_threads;

    /// The maximum possible number of threads that can be spawned
    int max_available_threads;

    /// Is multithreading enabled
    bool is_enabled;
};

RPY_NO_DISCARD RPY_EXPORT
ThreadState get_thread_state();

RPY_NO_DISCARD RPY_EXPORT
bool threading_available();

RPY_EXPORT
void set_num_threads(int num_threads);

RPY_EXPORT
void set_threading_enabled(bool state);


}}



#endif// ROUGHPY_PLATFORMS_THREADS_H
