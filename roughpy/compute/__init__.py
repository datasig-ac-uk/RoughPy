"""
Free-standing compute functions that operate on plain NumPy arrays.

This provides the internal computational routines used by RoughPy using a NumPy based interface.
The functions and classes provided here can be used to implement custom algorithms that
require operations on free tensors, shuffle tensors, and Lie algebras.

This interface is designed to be stable and efficient, and interoperable with RoughPy main objects,
but is not directly tied the internals of RoughPy itself.

Functions that operate on free tensor data specifically are prefixed with "ft" (e.g. `ft_fma`).
Functions that operate on shuffle tensors are prefixed with "st" (e.g. `st_shuffle`).

"""


# Bases

class TensorBasis:
    ...


class LieBasis:
    ...



# Basic functions

def ft_fma(*args, **kwargs):
    ...


def ft_inplace_mul(*args, **kwargs):
    ...


def st_fma(*args, **kwargs):
    ...


def st_inplace_mul(*args, **kwargs):
    ...



def lie_to_tensor(*args, **kwargs):
    ...


def tensor_to_lie(*args, **kwargs):
    ...


