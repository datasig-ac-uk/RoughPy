.. _shuffle_tensors:

***************
Shuffle Tensors
***************

^^^^^^^^^^^^^^^^^^^^^^^^
What are shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^

Shuffle tensors are one way of representing the linear functionals on free tensors.
The shuffle product corresponds to pointwise multiplication of the continuous functions on paths via the signature correspondence.
For more information on shuffle tensors, see :cite:t:`reutenauer_free_1993`.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do shuffle tensors fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shuffle tensors are useful because they represent functions on paths via the signature.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How to work with shuffle tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create shuffle tensors in the following way. In this example we will use polynomial coefficients to demonstrate how shuffles behave.

::

    >>> from roughpy import ShuffleTensor, Monomial

    >>> shuffle_tensor1 = ShuffleTensor([1 * Monomial(f"x{i}") for i in range(7)], width=2,
                    depth=2, dtype=roughpy.RationalPoly)
    >>> print(f"{shuffle_tensor1=!s}")
    shuffle_tensor1={ { 1(x0) }() { 1(x1) }(1) { 1(x2) }(2) { 1(x3) }(1,1) { 1(x4) }(1,2) { 1(x5) }(2,1) { 1(x6) }(2,2) }


    >>> shuffle_tensor2 = ShuffleTensor([1 * Monomial(f"y{i}") for i in range(7)], width=2,
                    depth=2, dtype=roughpy.RationalPoly)
    >>> print(f"{shuffle_tensor2=!s}")
    shuffle_tensor2={ { 1(y0) }() { 1(y1) }(1) { 1(y2) }(2) { 1(y3) }(1,1) { 1(y4) }(1,2) { 1(y5) }(2,1) { 1(y6) }(2,2) }

We can shuffle these together

::

    >>> result = shuffle_tensor1*shuffle_tensor2
    >>> print(f"{result=!s}")
    result={ { 1(x0 y0) }() { 1(x0 y1) 1(x1 y0) }(1) { 1(x0 y2) 1(x2 y0) }(2) { 1(x0 y3) 2(x1 y1) 1(x3 y0) }(1,1) { 1(x0 y4) 1(x1 y2) 1(x2 y1) 1(x4 y0) }(1,2) { 1(x0 y5) 1(x1 y2) 1(x2 y1) 1(x5 y0) }(2,1) { 1(x0 y6) 2(x2 y2) 1(x6 y0) }(2,2) }

^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::