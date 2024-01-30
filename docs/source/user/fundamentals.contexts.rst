.. _contexts:

**************
Contexts
**************

^^^^^^^^^^^^^^^^^^^^^
What are contexts
^^^^^^^^^^^^^^^^^^^^^

Contexts allow us to provide a Width, Depth, and a coefficient field for a tensor. They also provide access to the Baker-Campbell-Hausdorff formula.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do contexts fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contexts allow us to transfer between signatures and log signatures (t2l, l2t functions).

.. todo::
    - Finish contexts in RoughPy
    - Link transfer functions

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create **contexts** for :doc:`fundamentals.streams` in the following way. The example below has width 2, depth 6, and coefficients over the Reals.

::

    context = roughpy.get_context(width=2, depth=6, coeffs=rp.DPReal)

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::
