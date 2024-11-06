//
// Created by sammorley on 06/11/24.
//

#ifndef ROUGHPY_PYMODULE_PYTHON_H
#define ROUGHPY_PYMODULE_PYTHON_H

/*
 * Python header must be included first. This is a little wrapper that makes
 * sure the correct defines are made before including.
 *
 * Unfortunately, Python autolinks on Windows to the "correct" library version,
 * be that available or otherwise. This is really irritating. To get around this
 * We unsert _DEBUG before loading Python.
 */

#ifdef _DEBUG
#  define RPY_TMP_DEBUG _DEBUG
#endif
#undef _DEBUG

#define PY_SSIZE_T_CLEAN
#include <Python.h>


#endif //ROUGHPY_PYMODULE_PYTHON_H
