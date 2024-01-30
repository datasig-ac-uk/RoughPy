.. _shuffle_tensors:

***************
Shuffle Tensors
***************

^^^^^^^^^^^^^^^^^^^^^^^^
What are shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^

Shuffle tensors are

.. todo::
    Finish what are shuffle tensors

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do shuffle tensors fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will most commonly encounter shuffle tensors

.. todo::
    Finish shuffle tensors in RoughPy

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    from roughpy import ShuffleTensor

    d1 = tensor_data()
    shuffle_tensor1 = roughpy.ShuffleTensor(d1, ctx=tensor_context)

    shuffle_tensor2 = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2,
                    depth=2, dtype=roughpy.RationalPoly)

.. todo::

    - Need list of things to show
    - Shuffle two together, same as free tensor

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::