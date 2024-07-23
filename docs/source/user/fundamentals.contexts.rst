.. _contexts:

**************
Contexts
**************

^^^^^^^^^^^^^^^^^^^^^
What are contexts
^^^^^^^^^^^^^^^^^^^^^

Contexts allow us to provide a width, depth, and a coefficient field/type for a tensor. They also provide access to the Baker-Campbell-Hausdorff formula.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do contexts fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contexts are the environment in which calculations are done. They are used everywhere in RoughPy, for any stream or algebraic object.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create **contexts** for :doc:`fundamentals.streams` in the following way. The example below has width 2, depth 6, and coefficients over the Reals.

::

    context = roughpy.get_context(width=2, depth=6, coeffs=rp.DPReal)

