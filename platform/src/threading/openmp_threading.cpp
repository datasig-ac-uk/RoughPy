//
// Created by sam on 28/07/23.
//

#include <roughpy/platform/threads.h>

#include <atomic>

#include <omp.h>

#if !defined(RPY_THREADING_OPENMP) || !RPY_THREADING_OPENMP
#  error "OpenMP backend cannot be used unless OpenMP is avaiable."
#endif

using namespace rpy;
using namespace rpy::platform;

static const string s_omp_backend_name = "OpenMP";

static std::atomic_bool s_enable_omp_threading = true;

ThreadState rpy::platform::get_thread_state()
{
    return {ThreadBackend::OpenMP, omp_get_num_threads(), omp_get_max_threads(),
            s_enable_omp_threading.load(std::memory_order_relaxed)};
}


void rpy::platform::set_num_threads(int num_threads)
{
    omp_set_num_threads(num_threads);
}

void rpy::platform::set_threading_enabled(bool state)
{
    s_enable_omp_threading.store(state, std::memory_order_relaxed);
}
