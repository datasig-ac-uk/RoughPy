.. _shuffle_tensors:

***************
Shuffle Tensors
***************

^^^^^^^^^^^^^^^^^^^^^^^^
What are shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^

Shuffle tensors are one way of representing the linear functionals on free tensors.
The shuffle product corresponds to pointwise multiplication of the continuous functions on paths via the signature correspondence.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do shuffle tensors fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shuffle tensors are useful because they represent funcitons on paths via the signature.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    from roughpy import ShuffleTensor

    d1 = tensor_data()
    shuffle_tensor1 = roughpy.ShuffleTensor(d1, ctx=tensor_context)

    shuffle_tensor2 = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2,
                    depth=2, dtype=roughpy.RationalPoly)

We can shuffle these together

::

    result = rp.free_multiply(shuffle_tensor1, shuffle_tensor2)

.. todo::

    - Print output of shuffle

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::

.. todo::
    Include references: (create individual bib file)
        - Reutenauer