.. _lies:

**************
Lies
**************

^^^^^^^^^^^^^^^^^^^^^
What are Lie elements
^^^^^^^^^^^^^^^^^^^^^

Lie elements live in the free Lie Algebra. They have a one to one relationship with the group-like elements.

.. todo::
    - From notes "group like elts -> signatures". Ask??

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
How do Lies fit into RoughPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will most commonly encounter Lies when taking the log signature of a path. We can use the Dynkin map to transfer between Lies and signatures.

^^^^^^^^^^^^^^^^^^^^^
How to work with Lies
^^^^^^^^^^^^^^^^^^^^^

To create a Lie element, we should start by creating **data**.

::

    lie_data = [
    1 * roughpy.Monomial("x1"),  # channel (1)
    1 * roughpy.Monomial("x2"),  # channel (2)
    1 * roughpy.Monomial("x3"),  # channel (3)
    1 * roughpy.Monomial("x4"),  # channel ([1, 2])
    1 * roughpy.Monomial("x5"),  # channel ([1, 3])
    1 * roughpy.Monomial("x6"),  # channel ([2, 3])
    ]

Here ``roughpy.Monomial`` creates a monomial object (in this case, a single indeterminate),
and multiplying this by 1 makes this monomial into a polynomial.

We can then use our data to create a Lie element. Here we are creating two Lies with Width 3, Depth 2. The first is using a list of integers. The second is using our polynomial coefficients.

::

    lie1 = rp.Lie([1, 2, 3], width=3, depth=2)
    lie2 = rp.Lie(lie_data, width=3, depth=2, dtype=rp.RationalPoly)

We can also multply Lies together

::

    result = lie1*lie2

.. todo::
    Print multiply result


If we provide a complementary :doc:`fundamentals.contexts`, then we can use this to create a :doc:`fundamentals.free_tensors`, from our Lie element.

::

    context = rp.get_context(3, 2, rp.RationalPoly)

    tensor = context.lie_to_tensor(lie)

We can also go the other way, creating a Lie from a tensor. Using the signature of the stream shown ealier in :doc:`fundamentals.streams`, we can create a corresponding Lie element.

::

    lie = context.tensor_to_lie(sig)

.. todo::
    Print above result. Check: .log() or .exp()?? Start with a signature of a stream. Apply .log ctx tensor to lie. Add on end .exp()


^^^^^^^^^^^^^^^^^^^^^
Literature references
^^^^^^^^^^^^^^^^^^^^^

.. bibliography::

.. todo::

    Include references: (create individual bib file)
        - Reutenauer
        - Bourbaki Lie groups and Lie algebras
